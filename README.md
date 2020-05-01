# Renyi-Divergence-Variational-Inference

project description

## Getting Started

Follow these instructions to install the pre-requisites and execute the code

### Prerequisites

Install the following packages with their respective version to execute the code
```
torch 1.4.0
scipy 1.2.1
matplotlib 3.1.1
scikit-learn 0.22.2
```
### Variational Autoencoders
See README in the VAE folder.

### Bayesian Neural newtworks

The main code to replicate the experiments is the file `code.py` in the BNN folder <br />
The code to preprocess the data is `data.py` <br />
All of the datasets required for replication is present in `BNN/Datsets`<br />
Create a folder `Results`in the same folder where code is run <br />
The file `code.py` takes command line inputs for various parameters required to run experiments.
```
python code.py --dataset=dataset --alpha=alpha --seed=seed --lr=learning_rate --v_prior=v_prior --batch_size=batch_size --epochs=epochs --K=K --hidden_size=hidden layer size --offset=mean value --init_scale=standard deviation
```
All the experiments for replication of results presented in the report can be found in `jobs.sh`<br />
To execute this file run,<br />
```
./jobs.sh
```



