import torch
import numpy as np
import random
from tqdm import tqdm
import os
import copy
import argparse

from utils.data_loader import get_cifar10_dataloaders
from utils.logger import log_client_data_distribution, log_training_progress, log_weight_updates
from models.vgg import split_vgg11_for_cifar10
from client import Client
from server import Server

def main():
    """
    Main function to run the Split Federated Learning simulation.
    """
    parser = argparse.ArgumentParser(description="Run Split Federated Learning Simulation")
    parser.add_argument('--distribution', type=str, default='non-iid', choices=['iid', 'non-iid'],
                        help="Data distribution strategy (iid or non-iid)")
    parser.add_argument('--log_dir', type=str, default=None,
                        help="Directory to save logs. Default is 'logs_iid' or 'logs_non_iid'.")
    parser.add_argument('--dirichlet', type=float, default=0.5,
                        help="Alpha value for the Dirichlet distribution (for non-iid).")
    parser.add_argument('--gpu', type=int, default=0,
                        help="GPU device ID to use.")
    args = parser.parse_args()

    # --- 1. Hyperparameters and Setup ---
    print(f"--- 1. Setting up the experiment with {args.distribution.upper()} distribution ---")
    NUM_CLIENTS = 50
    CLIENTS_PER_ROUND = 5
    GLOBAL_ROUNDS = 100
    LOCAL_EPOCHS = 5
    LEARNING_RATE = 0.01
    BATCH_SIZE = 256
    DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    if args.log_dir:
        LOG_DIR = args.log_dir
    else:
        LOG_DIR = f"logs_{args.distribution}"

    # For reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # Create log directory
    os.makedirs(LOG_DIR, exist_ok=True)

    # --- 2. Data, Models, and Objects Initialization ---
    print("\n--- 2. Initializing data, models, and participants ---")
    # Prepare DataLoaders
    train_loaders, test_loader, client_indices = get_cifar10_dataloaders(
        num_clients=NUM_CLIENTS, 
        batch_size=BATCH_SIZE,
        distribution=args.distribution,
        alpha=args.dirichlet
    )
    
    # Log initial client data distribution
    log_client_data_distribution(client_indices, train_loaders[0].dataset.dataset, os.path.join(LOG_DIR, "client_data_distribution.json"))

    # Create global client and server models
    global_client_model, server_model = split_vgg11_for_cifar10(split_point=5)
    global_client_model.to(DEVICE)
    
    # Instantiate Server
    server = Server(server_model, test_loader, device=DEVICE, learning_rate=LEARNING_RATE)

    # Instantiate Clients
    clients = []
    for i in range(NUM_CLIENTS):
        client_model_copy = split_vgg11_for_cifar10(split_point=5)[0]
        client_model_copy.load_state_dict(global_client_model.state_dict())
        
        client = Client(
            client_id=i,
            train_loader=train_loaders[i],
            client_model=client_model_copy,
            device=DEVICE,
            learning_rate=LEARNING_RATE
        )
        clients.append(client)

    print(f"\nSetup complete. Starting {GLOBAL_ROUNDS} rounds of training with {CLIENTS_PER_ROUND}/{NUM_CLIENTS} clients per round.")

    # --- 3. Federated Training Loop ---
    for round_num in range(GLOBAL_ROUNDS):
        print(f"\n--- Global Round {round_num + 1}/{GLOBAL_ROUNDS} ---")
        
        weights_before = copy.deepcopy(global_client_model.state_dict())

        selected_client_ids = random.sample(range(NUM_CLIENTS), CLIENTS_PER_ROUND)
        print(f"Selected clients: {selected_client_ids}")
        
        round_losses = []
        local_model_weights = []

        for client_id in tqdm(selected_client_ids, desc="Local Training"):
            client = clients[client_id]
            client.model.to(DEVICE) # Move client model to GPU for training
            
            for epoch in range(LOCAL_EPOCHS):
                for data, labels in client.train_loader:
                    smashed_data = client.forward(data)
                    smashed_data_grad, loss = server.train_step(smashed_data, labels)
                    round_losses.append(loss)
                    client.backward(smashed_data, smashed_data_grad)
            
            local_model_weights.append(copy.deepcopy(client.get_model_weights()))
            client.model.to('cpu') # Move client model back to CPU to save memory

        global_client_weights = server.aggregate_models(local_model_weights)
        global_client_model.load_state_dict(global_client_weights)
        
        weights_after = copy.deepcopy(global_client_model.state_dict())

        for client in clients:
            client.set_model_weights(global_client_weights)
            
        accuracy = server.evaluate(global_client_model)
        avg_loss = sum(round_losses) / len(round_losses)
        
        print(f"Round {round_num + 1} Summary: Average Loss = {avg_loss:.4f}, Global Accuracy = {accuracy:.2f}%")

        log_entry = {
            "round": round_num + 1,
            "selected_clients": selected_client_ids,
            "average_loss": avg_loss,
            "global_accuracy": accuracy
        }
        log_training_progress(log_entry, os.path.join(LOG_DIR, "training_progress.json"))
        
        log_weight_updates(weights_before, weights_after, round_num + 1, LOG_DIR)

    print("\n--- Federated Training Finished ---")
    final_accuracy = server.evaluate(global_client_model)
    print(f"Final Global Model Accuracy: {final_accuracy:.2f}%")

if __name__ == '__main__':
    main()
