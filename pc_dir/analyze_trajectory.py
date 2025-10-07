import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os
import argparse

def get_top_class_for_clients(client_ids, client_dist):
    """Helper function to find the most dominant class for a list of clients."""
    top_classes = []
    for cid in client_ids:
        client_name = f"client_{cid}"
        if client_name in client_dist:
            dist = client_dist[client_name]
            top_class = max(dist, key=dist.get)
            top_classes.append(top_class)
    return ", ".join(top_classes)

def plot_update_vector_trajectory(log_path, save_dir):
    """
    Visualizes the trajectory of global model updates in a 2D space using PCA.
    This helps to show the instability of convergence in Non-IID settings.
    """
    print("--- Plotting model update trajectory using PCA ---")
    
    weights_dir = os.path.join(log_path, "weights")
    progress_filepath = os.path.join(log_path, "training_progress.json")
    dist_filepath = os.path.join(log_path, "client_data_distribution.json")

    if not all(os.path.exists(p) for p in [weights_dir, progress_filepath, dist_filepath]):
        print("Error: Necessary log files (weights, progress, distribution) not found.")
        return

    # 1. Load all necessary data
    with open(progress_filepath, 'r') as f:
        progress_data = [json.loads(line) for line in f]
    
    with open(dist_filepath, 'r') as f:
        client_dist_data = json.load(f)

    rounds = sorted([info['round'] for info in progress_data])
    if not rounds:
        print("No rounds found in progress log.")
        return

    # Load all weight vectors and calculate update vectors
    initial_weights_path = os.path.join(weights_dir, f"round_{rounds[0]}_before.pt")
    if not os.path.exists(initial_weights_path):
        print(f"Initial weight file not found: {initial_weights_path}")
        return
        
    initial_vec = torch.load(initial_weights_path, map_location='cpu')['features.0.weight'].flatten()
    
    update_vectors = []
    for r in rounds:
        path_before = os.path.join(weights_dir, f"round_{r}_before.pt")
        path_after = os.path.join(weights_dir, f"round_{r}_after.pt")
        if os.path.exists(path_before) and os.path.exists(path_after):
            vec_before = torch.load(path_before, map_location='cpu')['features.0.weight'].flatten()
            vec_after = torch.load(path_after, map_location='cpu')['features.0.weight'].flatten()
            update_vectors.append((vec_after - vec_before).numpy())

    if not update_vectors:
        print("No weight updates found to analyze.")
        return

    # 2. Fit PCA on the update vectors
    pca = PCA(n_components=2)
    pca.fit(update_vectors)

    # 3. Plot the trajectory
    plt.figure(figsize=(16, 12))
    ax = plt.gca()
    
    current_pos = np.zeros(2) # Start at origin
    
    for i, r in enumerate(rounds):
        round_info = progress_data[i]
        
        # Transform the update vector to 2D
        update_vec_2d = pca.transform([update_vectors[i]])[0]
        
        # Get info about selected clients
        selected_clients = round_info['selected_clients']
        top_classes = get_top_class_for_clients(selected_clients, client_dist_data)
        
        # Plot the arrow (update vector)
        ax.arrow(current_pos[0], current_pos[1], update_vec_2d[0], update_vec_2d[1],
                 head_width=0.01, head_length=0.01, fc=plt.cm.viridis(i / len(rounds)), ec=plt.cm.viridis(i / len(rounds)),
                 length_includes_head=True)
        
        # Update position for the next arrow
        next_pos = current_pos + update_vec_2d
        
        # Annotate the end of the arrow
        ax.text(next_pos[0] * 1.05, next_pos[1] * 1.05, 
                f"R{r}\nClients: {selected_clients}\n({top_classes})", 
                fontsize=9, ha='left', va='center')
        
        current_pos = next_pos

    plt.title('2D Trajectory of Global Model Updates (via PCA)')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid(True)
    ax.axhline(0, color='grey', lw=0.5)
    ax.axvline(0, color='grey', lw=0.5)
    
    # Set aspect ratio to be equal
    ax.set_aspect('equal', adjustable='box')

    save_path = os.path.join(save_dir, 'model_update_trajectory.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved trajectory plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze the trajectory of model updates from FL logs.")
    parser.add_argument("--log_path", type=str, default="../logs", help="Path to the logs directory.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save the output plots.")
    args = parser.parse_args()

    if not os.path.isdir(args.log_path):
        print(f"Error: Log directory not found at '{args.log_path}'")
        return
        
    os.makedirs(args.save_dir, exist_ok=True)

    plot_update_vector_trajectory(args.log_path, args.save_dir)

if __name__ == '__main__':
    main()
