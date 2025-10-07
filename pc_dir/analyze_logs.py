import json
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse

def plot_client_data_distribution(log_path, save_dir):
    """
    Plots a heatmap of the data distribution across all clients.
    """
    print("--- Plotting client data distribution heatmap ---")
    filepath = os.path.join(log_path, "client_data_distribution.json")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return

    with open(filepath, 'r') as f:
        data = json.load(f)
    
    df = pd.DataFrame(data).T.sort_index()
    df = df.fillna(0).astype(int)

    plt.figure(figsize=(20, 12))
    sns.heatmap(df, annot=True, fmt='d', cmap='viridis')
    plt.title('Data Distribution Across All Clients')
    plt.xlabel('Class')
    plt.ylabel('Client ID')
    
    save_path = os.path.join(save_dir, 'summary_client_data_distribution.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def plot_training_progress(log_path, save_dir):
    """
    Plots the training progress (accuracy and loss) over global rounds.
    """
    print("\n--- Plotting training progress (Accuracy & Loss) ---")
    filepath = os.path.join(log_path, "training_progress.json")
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return
        
    log_list = []
    with open(filepath, 'r') as f:
        for line in f:
            log_list.append(json.loads(line))
    
    df = pd.DataFrame(log_list)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Global Round')
    ax1.set_ylabel('Global Accuracy (%)', color=color)
    ax1.plot(df['round'], df['global_accuracy'], color=color, marker='o', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Average Loss', color=color)
    ax2.plot(df['round'], df['average_loss'], color=color, marker='x', linestyle='--', label='Loss')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.suptitle('Training Progress: Accuracy and Loss')
    fig.tight_layout()
    
    save_path = os.path.join(save_dir, 'summary_training_progress.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

def plot_round_summary(round_info, all_client_dist, log_path, save_dir, layer_name='features.0.weight'):
    """
    Generates a comprehensive summary plot for a single round, showing
    data distribution of selected clients and the resulting weight update.
    """
    round_num = round_info['round']
    print(f"\n--- Generating summary plot for Round {round_num} ---")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 8))
    fig.suptitle(f'Comprehensive Summary for Global Round {round_num}', fontsize=16)

    # --- Panel 1: Data Distribution of Selected Clients ---
    selected_clients = [f"client_{cid}" for cid in round_info['selected_clients']]
    round_dist_df = pd.DataFrame(all_client_dist).T.loc[selected_clients]
    
    round_dist_df.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
    ax1.set_title(f'Data Distribution of Selected Clients')
    ax1.set_xlabel('Client ID')
    ax1.set_ylabel('Number of Samples')
    ax1.tick_params(axis='x', rotation=0)
    ax1.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(axis='y')

    # --- Panel 2: Weight Update Histogram ---
    weights_dir = os.path.join(log_path, "weights")
    path_before = os.path.join(weights_dir, f"round_{round_num}_before.pt")
    path_after = os.path.join(weights_dir, f"round_{round_num}_after.pt")

    if not (os.path.exists(path_before) and os.path.exists(path_after)):
        ax2.text(0.5, 0.5, f"Weight files for round {round_num} not found", ha='center', va='center')
    else:
        weights_before = torch.load(path_before, map_location='cpu')
        weights_after = torch.load(path_after, map_location='cpu')

        if layer_name in weights_before and layer_name in weights_after:
            wb = weights_before[layer_name].flatten().numpy()
            wa = weights_after[layer_name].flatten().numpy()
            
            ax2.hist(wb, bins=100, alpha=0.7, label='Before Aggregation', color='gray')
            ax2.hist(wa, bins=100, alpha=0.7, label='After Aggregation', color='orange')
            ax2.set_title(f'Weight Distribution of "{layer_name}"')
            ax2.set_xlabel('Weight Value')
            ax2.set_ylabel('Frequency')
            ax2.legend()
            ax2.grid(True)
        else:
            ax2.text(0.5, 0.5, f"Layer '{layer_name}' not found", ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = os.path.join(save_dir, f'round_{round_num}_summary.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize logs from a federated learning simulation.")
    parser.add_argument("--log_path", type=str, default="../logs", help="Path to the logs directory generated by the server.")
    parser.add_argument("--save_dir", type=str, default="plots", help="Directory to save the output plots.")
    args = parser.parse_args()

    if not os.path.isdir(args.log_path):
        print(f"Error: Log directory not found at '{args.log_path}'")
        return
        
    os.makedirs(args.save_dir, exist_ok=True)

    # --- Generate Summary Plots ---
    plot_client_data_distribution(args.log_path, args.save_dir)
    plot_training_progress(args.log_path, args.save_dir)

    # --- Generate Per-Round Comprehensive Plots ---
    dist_filepath = os.path.join(args.log_path, "client_data_distribution.json")
    progress_filepath = os.path.join(args.log_path, "training_progress.json")

    if os.path.exists(dist_filepath) and os.path.exists(progress_filepath):
        with open(dist_filepath, 'r') as f:
            all_client_dist = json.load(f)
        
        with open(progress_filepath, 'r') as f:
            for line in f:
                round_info = json.loads(line)
                plot_round_summary(round_info, all_client_dist, args.log_path, args.save_dir)
    else:
        print("\nCould not generate per-round plots because log files were not found.")

if __name__ == '__main__':
    main()
