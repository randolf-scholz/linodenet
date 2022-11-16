from scipy.sparse import data
from linodenet.models import LinODEnet, filters, system, ResNet
from linodenet.models.filters import KalmanCell
from linodenet import projections

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import odeint, solve_ivp
from functools import partial
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import gen_batches
from scipy.stats import expon
import matplotlib.pyplot as plt
import matplotlib
from torch.utils.tensorboard import SummaryWriter
from tsdm.logutils import StandardLogger
import os
import argparse


matplotlib.use('agg')

def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    rgba = np.asarray(fig.canvas.buffer_rgba())
    #w,h = fig.canvas.get_width_height()
    #buf = np.fromstring ( fig.canvas.tostring_rgb(), 'uint8')
    #buf.shape = ( w, h,3 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    #buf = np.roll ( buf, 3, axis = 2 )
    return rgba


def vl(t,v,**kwargs):
    x,y = v
    return np.array([kwargs['alpha']*x-kwargs['beta']*x*y,-kwargs['gamma']*y + kwargs['delta']*x*y ])



def create_samples_from_volterra_lotka(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.3,v0=np.array([1.,1.])):

    times =  np.linspace(from_time,to_time,n_times)
    results = solve_ivp(partial(vl,**locals()),(from_time,to_time),v0,
    t_eval=times, method='LSODA')
    v = results['y']
    v[:,:] = np.where(np.random.rand(*v[:,:].shape)<freq_nan, np.nan*np.ones_like(v[:,:]),v[:,:])
    return results['t'],v


def create_dataset_from_one_system(n=100,kwargs = dict(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.3)):
    data_x = []
    data_t = []
    for i in range(n):
        v0 = np.random.rand(2)+0.5
        kwargs['v0'] = v0
        t,v = create_samples_from_volterra_lotka(**kwargs)
        data_t.append(t)
        data_x.append(v.T)
    return np.array(data_t), np.array(data_x)


def create_dataset_from_many_systems(n=1000):
    data_x = []
    data_t = []
    for i in range(n):
        alpha = 0.5*expon.rvs()+0.5
        beta = 0.5*expon.rvs()+0.5
        gamma = 0.5*expon.rvs()+0.5
        delta = 0.5*expon.rvs()+0.5
        #print(alpha,beta)
        kwargs = dict(alpha=alpha, beta=beta,gamma=gamma, delta=delta, from_time=0., to_time=30., n_times =300,freq_nan=0.0) 
        t,v = create_dataset_from_one_system(1, kwargs)
        data_t.append(t[0])
        data_x.append(v[0])
    return np.array(data_t,dtype=np.float), np.array(data_x,dtype=np.float)


def MSE_NAN(y,yhat):
    squares = torch.square(y-yhat) 
    loss = torch.mean (squares[~squares.isnan()])
    return loss

LOSS = torch.jit.script(MSE_NAN)

if __name__=="__main__":


    
   


    times, results = create_samples_from_volterra_lotka()
    kwargs = dict(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.0)

    data_t, data_x = create_dataset_from_many_systems(10000)#,kwargs)
    HIDDEN_SIZE = 0
    DTYPE = torch.float32
    DEVICE = 'cuda:0'
    PAST = 150
    BATCH_SIZE = 50
    
    rs = ShuffleSplit(n_splits=5, test_size=0.2)
    
    models = []
    
    HP = {
    "Filter": filters.SequentialFilter.HP | {"autoregressive": True},
    "System": system.LinODECell.HP | {"kernel_initialization": 'skew-symmetric'}, 
    "Encoder": ResNet.HP | {"num_blocks":4}
    }
   # "kernel_parametrization":partial(projections.functional.banded,l=-3,u=3)},
   # }

    DIM = 2
    LATENT = 64
    HIDDEN = 8

    directory = f"./.logs/r8_{LATENT}_{HIDDEN}/"

    for fold,(train_index, test_index) in enumerate(rs.split(data_t)):
        writer = SummaryWriter(directory)
        #writer.add_hparams(HP,{})
        last_test_loss = 1e20
        counts = 0
        model = LinODEnet(DIM,LATENT,HIDDEN, **HP).to(DEVICE)
        t_train = torch.Tensor(data_t[train_index]).type(DTYPE).to(DEVICE)
        x_train = torch.Tensor(data_x[train_index]).type(DTYPE).to(DEVICE)
        x_train_past = torch.ones_like(x_train).copy_(x_train).to(DEVICE)
        past = np.random.randint(PAST-50, PAST+50)
        x_train_past[:,past:,:] = torch.nan
        t_test = torch.Tensor(data_t[test_index]).type(DTYPE).to(DEVICE)
        x_test = torch.Tensor(data_x[test_index]).type(DTYPE).to(DEVICE)
        x_test_past = torch.ones_like(x_test).copy_(x_test).to(DEVICE)
        x_test_past[:,past:,:] = torch.nan
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-3)
        for epoch in range(1000):
            train_losses = []
            for i,batch in enumerate(gen_batches(len(x_train),BATCH_SIZE)):
                optimizer.zero_grad()
                out = model(t_train[batch], x_train_past[batch])
                loss = LOSS(x_train[batch,:,:], out)
                #squares = torch.square(out[,:,:]- x_train[batch,:,:]) 
                #loss = torch.mean (squares[~squares.isnan()])
                loss.backward()
                optimizer.step()
                print(epoch,i,loss.item())
                train_losses.append(loss.item())
                #print([t.grad for t in model.parameters()])
            
            print("Av. train loss", np.mean(train_losses))

            out = model(t_test, x_test_past)
            
            for j in range(50):

                fig = plt.figure( )
                i = np.random.randint(0, x_test.shape[0])
                example = out[i].detach().to('cpu').numpy()
                example_gt = x_test[i].detach().to('cpu').numpy()
                example_t = t_test[i].detach().to('cpu').numpy()
                plt.plot(example_t,  example[:,0],color='blue')
                plt.plot(example_t, example[:,1],color='red')
                plt.plot(example_t,  example_gt[:,0],color='blue',linestyle='dashed')
                plt.plot(example_t, example_gt[:,1],color='red', linestyle='dashed')
                image_array = fig2data(fig)
                plt.close()
                writer.add_image(f'img/{epoch}_{i}', image_array,dataformats='HWC')
            squares = torch.square(out[:,:,:]- x_test[:,:,:]) 
            loss = torch.mean (squares[~squares.isnan()])
        
            print("Test: ", epoch,loss.item())
            writer.add_scalar(f'Train/loss fold{fold}',np.mean(train_losses),epoch)
            writer.add_scalar(f'Test/loss fold{fold}', loss.item(),epoch)


            if loss.item()>last_test_loss:
                counts +=1
                if counts > 3:
                    print("Early stopping")
                    break
            else:
                counts = 0
                torch.save(model.state_dict(), os.path.join(directory,f'checkpoint_{fold}.torch'))
            last_test_loss = loss.item()
        models.append(model)



    

    


