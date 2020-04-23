# GreNADE

GreNADE is for quantum state reconstruction using Neural Autoregressive Distribution Estimators (NADEs).

### Usage

Firstly, include ```NADE.jl```.

```julia
include("NADE.jl")
```

```NADE.jl``` includes all relevant functions to train a NADE. However, the user must specify the following.

- ```train_data``` : a file containing samples of binary data
- ```Nh``` : the number of hidden units

Now, to initialize the NADE parameters, call the ```initialize_parameters()``` function. There are two keyword arguments for this:

- ```seed```: (default: 1234) the random seed for initializing the NADE weights
- ```zero_weights```: (Bool, default: false) choice of initializing the NADE weights to zero or not. Of course, this will override the seed if it was specified. So, ```initialize_parameters(seed=9999, zero_weights=true)``` won't do anything with ```seed``` and one could have equivalently called ```initialize_parameters(zero_weights=true)```.

The biases of the NADE are always set to initialize to zero. Now, specify what is needed to call the ```train``` function to train the NADE.

    train_data;
    batch_size=100, 
    opt=ADAM(), 
    epochs=1000, 
    log_every=100,
    calc_fidelity=false,
    target=nothing,
    calc_observable=false,
    num_samples=nothing,
    observable=nothing,
    early_stopping=nothing,
    early_stopping_args=nothing

- ```train_data```: binary input data
- ```batch_size```: (integer, default: 100) the mini batch size used for calculating gradients
- ```opt```: the optimization method (e.g. ```ADAM()```). These are optimizers available in Flux.
- ```epochs```: (integer, default:1000) number of training steps (passes through the input data)
- ```calc_fidelity```: (Bool, default: ```false```) Do you want to monitor the fidelity while training the NADE?
- ```target```: The target quantum state. If ```calc_fidelity=true```, this is required (of course!)
- ```calc_observable```: (Bool, default: ```false```) Do you want to monitor an observable while training the NADE?
- ```num_samples```: (integer, default: ```nothing```) if ```calc_observable=true```, then we need to know how many samples you want to generate from the NADE to calculate your observable on.
- ```observable```: (function, default: ```nothing```, returns: the value of the observable on one sample) This is a user-specified function that calculates the value of an observable given one sample from the NADE. 
- ```log_every```: the frequeny (in epochs) that one wishes to monitor their training metric (fidelity or an observable)
- ```early_stopping```: (function, default: ```nothing```, returns: Bool) This is a user-specified function that defines a learning criteria for the NADE that, once met during the training, stops the training early (i.e. before the last epoch). The arguments to this function must be: the "current" metric value (e.g. if you're calculating fidelity, you must input the current fidelity in the training process) and other arguments required (see ```early_stopping_args```)
- ```early_stopping_args```: Other required arguments required for the ```early_stopping``` function. 

If you're at all confused, see ```run.jl``` for an example of how to train a NADE.