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
- the NADE parameters: ```W``` (```Nh``` x ```N```), ```U``` (```N``` x ```Nh```), ```b``` (length ```N```), ```c``` (length ```Nh```)

Then, pass these parameters into a tuple (in this order).

```julia
θ = (b, c, U, W)
```

Now, there are default hyperparameters in the NADE, but they can be changed. See the ```train``` function in ```NADE.jl```.

Currently, one can monitor fidelity and / or an observable during the training process. At the end of training, whatever was chosen to be monitored during training will be saved along with the NADE parameters.

See ```run.jl``` for an example.