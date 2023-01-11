# TODOs

## LinODEnet

### TODO: Control variables

xhat = self.control(xhat, u)
u: possible controls:

1.  set to value
2.  add to value
    do these via indicator variable
    u = (time, value, mode-indicator, col-indicator)
    => apply control to specific column.

### TODO: Smarter initialization

IDEA: The problem is the initial state of RNNCell is not defined and typically put equal
to zero. Staying with the idea that the Cell acts as a filter, that is updates the state
estimation given an observation, we could "trust" the original observation in the sense
that we solve the fixed point equation h0 = g(x0, h0) and put the solution as the initial
state.
issue: if x0 is really sparse this is useless.
better idea: we probably should go back and forth.
other idea: use a set-based model and put h = g(T,X), including the whole TS.
This set model can use triplet notation.
bias weighting towards close time points

### TODO: replace with add_module once supported!

```python
self.add_module("embedding", _embedding)
self.add_module("encoder", HP["Encoder"](**HP["Encoder_cfg"]))
self.add_module("system", HP["System"](**HP["System_cfg"]))
self.add_module("decoder", HP["Decoder"](**HP["Decoder_cfg"]))
self.add_module("projection", _projection)
self.add_module("filter", HP["Filter"](**HP["Filter_cfg"]))
```
