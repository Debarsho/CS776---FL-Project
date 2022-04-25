# Dynamic Logit-based Clustering in Federated Learning

CS776: Group 6: Foresight (Ankur, Debarsho Sannyasi, Priyanshu Agarwal, Yuvraj)

## Implementation points
* Exactly implement the non-IID data split.
* Implement multiprocessing of _client update_ and _client evaluation_.
* Support TensorBoard for log tracking.

## Requirements
* See `requirements.txt`

## Configurations
* See `config.yaml`

## Run
* `python3 main.py`

## Configuration file details (`config.yaml`)
* Number of clients: K (K = 100)
* Fraction of sampled clients: C
* Number of rounds: R
* Number of local epochs: E (E = 50)
* Batch size: B (B = 20)
* Number of Clusters: NC (NC=4)
* Optimizer: `torch.optim.SGD`
* Criterion: `torch.nn.CrossEntropyLoss`
* Learning rate: 0.01
* Momentum: 0.9
* Initialization: Xavier

## src directory details
* `client.py`: Defines all the client side activites, like client model training, evaluation, etc.
* `server.py`: Defines all the SERVER side activites, like the clustering algorithm, training and evaluation of the global models, setting up the server and the client models, etc.
* `models.py`: Defines all the model architectures. We have mainly used CNN2. 
* `utils.py`: Consists of various utility functions such as creation of various datasets, initializing the models, etc.

### Choose the appropriate clustering algorithm in cluster() function in `src/server.py`


