from scipy.sparse import data
from linodenet.models import LinODEnet, filters, system, ResNet
from linodenet.models.filters import KalmanCell, SequentialFilter, PseudoKalmanFilter
from linodenet.models.filters._filters import  LinearFilter, NonLinearFilter, NonLinearFilter2
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
#from tsdm.logutils import StandardLogger
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



def create_samples_from_volterra_lotka(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.3,v0=np.array([1.,1.]), rel_noise=0.):

    times =  np.linspace(from_time,to_time,n_times)
    results = solve_ivp(partial(vl,**locals()),(from_time,to_time),v0,
    t_eval=times, method='LSODA')
    v = results['y']
    v = np.exp(np.log(v) + rel_noise*np.random.randn(*v.shape))
    v[:,:] = np.where(np.random.rand(*v[:,:].shape)<freq_nan,np.nan*np.ones_like(v[:,:]),v[:,:])

    return results['t'],v


def create_dataset_from_one_system(n=100,kwargs = dict(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.3, rel_noise=0.)):
    data_x = []
    data_t = []
    for i in range(n):
        v0 = np.random.rand(2)+0.5
        kwargs['v0'] = v0
        t,v = create_samples_from_volterra_lotka(**kwargs)
        data_t.append(t)
        data_x.append(v.T)
    return np.array(data_t), np.array(data_x)


def create_dataset_from_many_systems(n=1000, freq_nan=0.0, n_times=300, from_time=0, to_time=30,rel_noise=0.):
    data_x = []
    data_t = []
    for i in range(n):
        alpha = 0.5*expon.rvs()+0.5
        beta = 0.5*expon.rvs()+0.5
        gamma = 0.5*expon.rvs()+0.5
        delta = 0.5*expon.rvs()+0.5
        #print(alpha,beta)
        kwargs = dict(alpha=alpha, beta=beta,gamma=gamma, delta=delta, from_time=from_time, to_time=to_time, n_times = n_times,freq_nan=freq_nan,rel_noise=rel_noise) 
        t,v = create_dataset_from_one_system(1, kwargs)
        data_t.append(t[0])
        data_x.append(v[0])
    return np.array(data_t,dtype=float), np.array(data_x,dtype=float)


def MSE_NAN(y,yhat):
    positions = ~y.isnan()
    squares = torch.square(y[positions]-yhat[positions]) 
    
    loss = torch.mean (squares)
    return loss

def MSE_NAN_TO_ZERO(y,yhat):
    squares = torch.nan_to_num(torch.square(y-yhat)) 
    loss = torch.mean (squares)
    return loss

LOSS = torch.jit.script(MSE_NAN)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--HIDDEN",type=int, default=8)
    parser.add_argument("--LATENT", type=int, default=64)
    parser.add_argument("--DIM", type=int, default=2)
    parser.add_argument("--PAST", type=int, default=150)
    parser.add_argument("--DEVICE", type=str, default='cpu')
    parser.add_argument("--BATCH_SIZE",type=int, default=50)
    parser.add_argument("--EXPERIMENT_NAME", type=str,  default = 'r1')
    parser.add_argument("--PATIENCE",type=int, default=5)
    parser.add_argument("--FREQ_NAN",type=float, default=0.0)
    parser.add_argument("--REL_NOISE",type=float, default=0.01)
    parser.add_argument("--FILTER_TYPE", type=int, default=0)

    try:
        args = parser.parse_args()
    except:
        print("No arguments passed -- use default")
        args = parser.parse_args('')
    globals().update(args.__dict__)
   


    times, results = create_samples_from_volterra_lotka()
    kwargs = dict(alpha=0.66, beta=1.33,gamma=1., delta=1., from_time=0., to_time=30., n_times =300,freq_nan=0.0)

    data_t, data_x = create_dataset_from_many_systems(10000, freq_nan=FREQ_NAN, rel_noise=REL_NOISE)#,kwargs)

    DTYPE = torch.float32
    #DEVICE = 'cpu'
    #PAST = 150
    #BATCH_SIZE = 
    
    rs = ShuffleSplit(n_splits=5, test_size=0.2)
    
    models = []
    
    HP = {
    "Filter": filters.SequentialFilter.HP | {"autoregressive": True},
    "System": system.LinODECell.HP | {"kernel_initialization": 'skew-symmetric'}, 
    "Encoder": ResNet.HP | {"num_blocks":5}
    }

    
    if FILTER_TYPE==1:
        HP["Filter"]["layers"] = [LinearFilter.HP, NonLinearFilter2.HP, NonLinearFilter2.HP]
    elif FILTER_TYPE==2:
        HP["Filter"]["layers"] = [LinearFilter.HP, NonLinearFilter.HP]
    elif FILTER_TYPE==3:
        HP["Filter"]["layers"] = [LinearFilter.HP, NonLinearFilter.HP,NonLinearFilter.HP,NonLinearFilter.HP]
    elif FILTER_TYPE==4:
        nonlinear_filter_hp = NonlinearFilter.HP | {"num_blocks":6}
        HP["Filter"]["layers"] = [LinearFilter.HP, nonlinear_filter_hp]

   # "kernel_parametrization":partial(projections.functional.banded,l=-3,u=3)},
   # }
    #DIM = 2
    #LATENT = 64
    #HIDDEN = 8

    directory = f"./.logs_2/{EXPERIMENT_NAME}_{LATENT}_{HIDDEN}_{FILTER_TYPE}/"

    npnan = torch.tensor(np.nan) 

    for fold,(train_index, test_index) in enumerate(rs.split(data_t)):
        writer = SummaryWriter(directory)
        
        last_test_loss = 1e20
        counts = 0
        model = LinODEnet(DIM,LATENT,HIDDEN, **HP).to(DEVICE)
        model = torch.jit.script(model)
        t_train = torch.from_numpy(data_t[train_index]).type(DTYPE).to(DEVICE)
        x_train = torch.from_numpy(data_x[train_index]).type(DTYPE).to(DEVICE)
        x_train = torch.ones_like(x_train).copy_(x_train).to(DEVICE)
        #x_train[x_train.isnan()] = float("nan")
        x_train_past = torch.ones_like(x_train).copy_(x_train).to(DEVICE)
        past =  PAST
        x_train_past[:,past:,:] = torch.nan
        
        t_test = torch.from_numpy(data_t[test_index]).type(DTYPE).to(DEVICE)
        x_test = torch.from_numpy(data_x[test_index]).type(DTYPE).to(DEVICE)
        x_test_past = torch.ones_like(x_test).copy_(x_test).to(DEVICE)
        x_test_past[:,past:,:] = torch.nan
        optimizer = torch.optim.Adam(model.parameters(),lr=3e-3)
        for epoch in range(1000):
            train_losses = []
            reshuffling = np.random.permutation(list(range(x_train.shape[0])))
            x_train = x_train[reshuffling]
            x_train_past = x_train_past[reshuffling]
            t_train = t_train[reshuffling]
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
            with torch.no_grad():
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
                writer.add_image(f'img/{fold}_{epoch}_{i}', image_array,dataformats='HWC')
            
            loss = LOSS(x_test[:,:,:], out[:,:,:])
            
            #squares = torch.square(out[:,:,:]- x_test[:,:,:]) 
            #loss = torch.mean (squares[~squares.isnan()])
        
            print("Test: ", epoch,loss.item())
            writer.add_scalar(f'Train/loss fold{fold}',np.mean(train_losses),epoch)
            writer.add_scalar(f'Test/loss fold{fold}', loss.item(),epoch)

            if np.isnan(loss.item()):
                break

            if loss.item()>last_test_loss:
                counts +=1
                if counts > PATIENCE:
                    print("Early stopping")
                    break
            else:
                counts = 0
                torch.jit.save(model, os.path.join(directory,f'checkpoint_{fold}.torch'))
                #torch.save(model.state_dict(), os.path.join(directory,f'checkpoint_{fold}.torch'))
                last_test_loss = loss.item()
        writer.add_hparams(HP|args.__dict__,{'train_loss': np.mean(train_losses), 'test loss':last_test_loss})
        models.append(model.to('cpu'))



    

    


