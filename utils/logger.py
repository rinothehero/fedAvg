import json
import torch
import os
import numpy as np

# Helper to convert numpy types to python native types for JSON serialization
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

def log_client_data_distribution(client_indices, train_dataset, filepath):
    """
    Logs the data distribution of all clients to a JSON file.

    Args:
        client_indices (dict): Dictionary mapping client_id to a list of data indices.
        train_dataset: The original training dataset object.
        filepath (str): Path to save the JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    num_classes = 10
    class_names = train_dataset.classes
    
    distribution = {}
    for client_id, indices in client_indices.items():
        labels = [train_dataset.targets[i] for i in indices]
        class_counts = {class_names[i]: 0 for i in range(num_classes)}
        for label in labels:
            class_counts[class_names[label]] += 1
        distribution[f"client_{client_id}"] = class_counts
        
    with open(filepath, 'w') as f:
        json.dump(distribution, f, indent=4, cls=NpEncoder)
    print(f"Client data distribution saved to {filepath}")

def log_training_progress(log_data, filepath):
    """
    Appends a log entry for the current training round to a JSON file.

    Args:
        log_data (dict): A dictionary containing the log information for the round.
        filepath (str): Path to the JSON log file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'a') as f:
        f.write(json.dumps(log_data, cls=NpEncoder) + '\n')

def log_weight_updates(weights_before, weights_after, round_num, log_dir):
    """
    Saves the model state_dicts before and after aggregation for a round.

    Args:
        weights_before (OrderedDict): Global client model state_dict before aggregation.
        weights_after (OrderedDict): Global client model state_dict after aggregation.
        round_num (int): The current global round number.
        log_dir (str): The directory to save the weight files.
    """
    weights_dir = os.path.join(log_dir, "weights")
    os.makedirs(weights_dir, exist_ok=True)
    
    path_before = os.path.join(weights_dir, f"round_{round_num}_before.pt")
    path_after = os.path.join(weights_dir, f"round_{round_num}_after.pt")
    
    torch.save(weights_before, path_before)
    torch.save(weights_after, path_after)

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Testing logger functions ---")
    
    # 1. Test log_client_data_distribution
    # Create dummy data
    dummy_indices = {
        0: [0, 1, 2, 10, 11],
        1: [3, 4, 5, 12, 13]
    }
    class DummyDataset:
        def __init__(self):
            self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            self.targets = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4] # Dummy labels
    
    log_client_data_distribution(dummy_indices, DummyDataset(), "logs/test_client_dist.json")
    
    # 2. Test log_training_progress
    log_entry = {
        "round": 1,
        "selected_clients": [0, 1],
        "avg_loss": 2.104,
        "global_accuracy": 15.23
    }
    log_training_progress(log_entry, "logs/test_progress.json")
    log_entry_2 = {
        "round": 2,
        "selected_clients": [2, 3],
        "avg_loss": 1.987,
        "global_accuracy": 18.91
    }
    log_training_progress(log_entry_2, "logs/test_progress.json")
    print("Training progress logged to logs/test_progress.json")

    # 3. Test log_weight_updates
    dummy_weights_before = {'layer1.weight': torch.randn(3, 3)}
    dummy_weights_after = {'layer1.weight': torch.randn(3, 3)}
    log_weight_updates(dummy_weights_before, dummy_weights_after, 0, "logs")
    print("Weight updates logged in logs/weights/")

    print("\nLogger test finished.")
