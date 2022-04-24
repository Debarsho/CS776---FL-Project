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

## Configuration file details
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

