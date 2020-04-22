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

The biases of the NADE are always set to initialize to zero. There are other default hyperparameters in the NADE and they can be changed. See the ```train``` function in ```NADE.jl```.

Currently, one can monitor fidelity and / or an observable during the training process. At the end of training, whatever was chosen to be monitored during training will be saved along with the NADE parameters. Early stopping is a feature that will be added soon. 

See ```run.jl``` for an example.
