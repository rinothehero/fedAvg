# Split Federated Learning with Federated Averaging

This project implements a Split Federated Learning (SFL) framework using the Federated Averaging (FedAvg) algorithm on the CIFAR-10 dataset. The model used is a VGG-11, split between clients and a central server.

## Features

- **Split Learning**: The VGG-11 model is divided into a client-side part and a server-side part. Clients perform the initial forward pass, send the intermediate "smashed data" to the server, and perform a backward pass after receiving gradients from the server.
- **Federated Averaging (FedAvg)**: The server aggregates the model weights from selected clients at the end of each round to create a new global model.
- **Federated Proximal (FedProx)**: Implements the FedProx algorithm to counteract statistical heterogeneity in Non-IID settings by adding a proximal term to the local client objective function.
- **IID and Non-IID Data Distribution**: Supports both Independent and Identically Distributed (IID) and Non-IID data partitioning among clients. The Non-IID distribution is simulated using a Dirichlet distribution, controlled by an `alpha` parameter.
- **Detailed Logging**: Logs training progress (loss, accuracy), client data distribution, and model weight updates for analysis.

## Project Structure

```
/
├───main.py             # Main script to run the SFL simulation
├───server.py           # Implements the Server logic
├───client.py           # Implements the Client logic
├───models/
│   └───vgg.py          # Defines and splits the VGG-11 model
├───utils/
│   ├───data_loader.py  # Handles CIFAR-10 data loading and partitioning
│   └───logger.py       # Utility for logging
├───data/               # (Git-ignored) Stores the CIFAR-10 dataset
└───logs/               # (Git-ignored) Stores experiment logs and weights
```

## Requirements

To run this simulation, you need to install the required Python packages. It is recommended to use a virtual environment.

First, check the requirements file:
```bash
cat pc_dir/requirements.txt
```

Then, install the packages:
```bash
pip install -r pc_dir/requirements.txt
```

## Usage

You can run the simulation using `main.py`.

### FedAvg with IID Distribution

To run the simulation with an IID data distribution across 50 clients:
```bash
python main.py --distribution iid --gpu 0
```

### FedAvg with Non-IID Distribution

To run with a Non-IID data distribution (controlled by the Dirichlet `alpha` parameter):
```bash
python main.py --distribution non-iid --dirichlet 0.5 --gpu 0
```

### FedProx with Non-IID Distribution

To run the simulation using the FedProx algorithm to mitigate the impact of Non-IID data, set the `--mu` parameter to a value greater than 0.
```bash
python main.py --distribution non-iid --dirichlet 0.1 --mu 0.1 --gpu 0
```

### Command-Line Arguments

- `--distribution`: Data distribution strategy. Choices: `iid`, `non-iid`. (Default: `non-iid`)
- `--log_dir`: Directory to save logs. (Default: `logs_iid` or `logs_non_iid_0.5`)
- `--dirichlet`: Alpha value for the Dirichlet distribution for Non-IID. (Default: `0.5`)
- `--gpu`: GPU device ID to use. (Default: `0`)
- `--mu`: The proximal term coefficient for FedProx. A value of `0.0` effectively runs FedAvg. (Default: `0.0`)
- `--clients_per_round`: Number of clients to select per round. (Default: `5`)
- `--global_rounds`: Total number of global training rounds. (Default: `100`)
