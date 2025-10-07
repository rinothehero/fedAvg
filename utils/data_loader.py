import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_cifar10_dataloaders(num_clients, batch_size, data_path='./data', distribution='non-iid', alpha=0.5):
    """
    Downloads CIFAR-10 and partitions it among clients.

    Args:
        num_clients (int): The total number of clients.
        batch_size (int): The batch size for the DataLoaders.
        data_path (str): Path to store/load the CIFAR-10 data.
        distribution (str): 'iid' or 'non-iid'.
        alpha (float): The concentration parameter for the Dirichlet distribution (for non-iid).

    Returns:
        tuple: A tuple containing train_loaders, test_loader, and client_indices.
    """
    print(f"[DEBUG] Starting get_cifar10_dataloaders with {distribution.upper()} distribution...")
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    train_dataset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

    client_indices = {i: [] for i in range(num_clients)}

    if distribution == 'iid':
        # --- IID Partitioning ---
        all_indices = list(range(len(train_dataset)))
        np.random.shuffle(all_indices)
        samples_per_client = len(train_dataset) // num_clients
        
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client
            client_indices[i] = all_indices[start_idx:end_idx]

    elif distribution == 'non-iid':
        # --- Non-IID Partitioning (Dirichlet) ---
        num_classes = 10
        labels = np.array(train_dataset.targets)
        class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
        
        for i in range(num_classes):
            idx = class_indices[i]
            np.random.shuffle(idx)
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            num_samples_per_client = (proportions * len(idx)).astype(int)
            diff = len(idx) - np.sum(num_samples_per_client)
            for j in range(diff):
                num_samples_per_client[j % num_clients] += 1
            start = 0
            for client_id, count in enumerate(num_samples_per_client):
                end = start + count
                client_indices[client_id].extend(idx[start:end])
                start = end
    else:
        raise ValueError("Distribution must be 'iid' or 'non-iid'.")

    print("[DEBUG] Data partitioning complete.")

    train_loaders = []
    for i in range(num_clients):
        subset = Subset(train_dataset, client_indices[i])
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_loaders.append(loader)
    print("[DEBUG] Train loaders created.")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size * 2, shuffle=False, num_workers=4, pin_memory=True)
    print("[DEBUG] Test loader created.")

    return train_loaders, test_loader, client_indices

if __name__ == '__main__':
    # --- Example Usage ---
    print("--- Testing IID distribution ---")
    train_loaders_iid, _, client_indices_iid = get_cifar10_dataloaders(50, 256, distribution='iid')
    if train_loaders_iid:
        print(f"Client 0 (IID) has {len(train_loaders_iid[0].dataset)} samples.")
        # Check class distribution for an IID client
        labels = [train_loaders_iid[0].dataset.dataset.targets[idx] for idx in client_indices_iid[0]]
        class_counts = {j: labels.count(j) for j in range(10)}
        print(f"  IID Class distribution: {class_counts}") # Should be roughly equal

    print("\n--- Testing Non-IID distribution ---")
    train_loaders_noniid, _, client_indices_noniid = get_cifar10_dataloaders(50, 256, distribution='non-iid')
    if train_loaders_noniid:
        print(f"Client 0 (Non-IID) has {len(train_loaders_noniid[0].dataset)} samples.")
        # Check class distribution for a Non-IID client
        labels = [train_loaders_noniid[0].dataset.dataset.targets[idx] for idx in client_indices_noniid[0]]
        class_counts = {j: labels.count(j) for j in range(10)}
        print(f"  Non-IID Class distribution: {class_counts}") # Should be skewed
