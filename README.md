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
* Number of clients: 100 (K = 100)
* Fraction of sampled clients: 0.1 (C = 0.1)
* Number of rounds: 500 (R = 500)
* Number of local epochs: 10 (E = 10)
* Batch size: 10 (B = 10)
* Optimizer: `torch.optim.SGD`
* Criterion: `torch.nn.CrossEntropyLoss`
* Learning rate: 0.01
* Momentum: 0.9
* Initialization: Xavier

