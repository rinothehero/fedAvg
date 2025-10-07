import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.vgg import ServerVGG, ClientVGG

class Server:
    """
    Represents the server in the Split Federated Learning setup.
    """
    def __init__(self, server_model, test_loader, device, learning_rate=0.01):
        """
        Initializes the Server object.

        Args:
            server_model (ServerVGG): The server-side part of the VGG model.
            test_loader (DataLoader): The DataLoader for the global test dataset.
            device (torch.device): The device to run the model on (e.g., 'cuda:0' or 'cpu').
            learning_rate (float): The learning rate for the optimizer.
        """
        self.device = device
        self.model = server_model.to(self.device)
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def aggregate_models(self, local_model_weights):
        """
        Aggregates the weights from multiple client models using Federated Averaging.

        Args:
            local_model_weights (list[dict]): A list of state dictionaries from the clients.

        Returns:
            dict: The new state dictionary for the global client-side model.
        """
        if not local_model_weights:
            return None
        
        # Initialize a dictionary to hold the averaged weights
        aggregated_weights = {}
        
        # Get the keys from the first model's state_dict
        keys = local_model_weights[0].keys()
        
        for key in keys:
            # Sum the weights for the current key from all client models
            # Move tensors to CPU for aggregation to avoid potential device mismatches
            summed_tensor = torch.stack([weights[key].cpu() for weights in local_model_weights]).sum(0)
            aggregated_weights[key] = summed_tensor / len(local_model_weights)
            
        return aggregated_weights

    def train_step(self, smashed_data, labels):
        """
        Performs one training step on the server-side model.

        Args:
            smashed_data (torch.Tensor): The intermediate output from a client model.
            labels (torch.Tensor): The true labels for the data batch.

        Returns:
            tuple: A tuple containing the gradients for the smashed_data and the loss value.
        """
        self.model.train()
        
        # Ensure labels are on the correct device
        labels = labels.to(self.device)
        
        # Detach smashed_data from the client's graph and require gradients for the server's graph
        smashed_data = smashed_data.detach().requires_grad_(True)
        
        # Clear previous gradients
        self.optimizer.zero_grad()
        
        # Forward pass through the server model
        outputs = self.model(smashed_data)
        loss = self.criterion(outputs, labels)
        
        # Calculate gradients for both server model parameters and smashed_data
        (smashed_data_grad, *server_grads) = torch.autograd.grad(
            outputs=loss, 
            inputs=[smashed_data] + list(self.model.parameters()),
            allow_unused=True
        )
        
        # Apply gradients to the server model parameters and update
        for param, grad in zip(self.model.parameters(), server_grads):
            if grad is not None:
                param.grad = grad
        self.optimizer.step()
        
        return smashed_data_grad, loss.item()

    def evaluate(self, global_client_model):
        """
        Evaluates the performance of the combined global model on the test dataset.

        Args:
            global_client_model (ClientVGG): The current global client-side model.

        Returns:
            float: The accuracy of the model on the test dataset.
        """
        self.model.eval()
        global_client_model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, labels in self.test_loader:
                data, labels = data.to(self.device), labels.to(self.device)
                
                # Pass data through both client and server models
                smashed_data = global_client_model(data)
                outputs = self.model(smashed_data)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        accuracy = 100 * correct / total
        return accuracy

if __name__ == '__main__':
    # --- Test the Server class functionality ---
    from models.vgg import split_vgg11_for_cifar10
    from utils.data_loader import get_cifar10_dataloaders

    print("--- Initializing a test server ---")
    
    # 1. Prepare dummy data and models
    _, test_loader, _ = get_cifar10_dataloaders(num_clients=1, batch_size=256)
    client_net, server_net = split_vgg11_for_cifar10()

    # 2. Create a Server instance
    server = Server(server_model=server_net, test_loader=test_loader)
    print(f"Server initialized on device: {server.device}")

    # 3. Simulate a single training step
    print("\n--- Simulating a single training step ---")
    try:
        # Create dummy smashed_data from a client
        dummy_smashed_data = torch.randn(256, 512, 2, 2).to(server.device)
        dummy_labels = torch.randint(0, 10, (256,)).to(server.device)
        
        smashed_grad, loss = server.train_step(dummy_smashed_data, dummy_labels)
        print(f"Server train step completed. Loss: {loss:.4f}")
        print(f"Returned gradient for smashed_data has shape: {smashed_grad.shape}")

    except Exception as e:
        print(f"\nAn error occurred during the train step test: {e}")

    # 4. Test model evaluation
    print("\n--- Testing model evaluation ---")
    try:
        # Move client model to the same device for evaluation
        client_net.to(server.device)
        accuracy = server.evaluate(client_net)
        print(f"Evaluation completed. Initial accuracy: {accuracy:.2f}%")
    
    except Exception as e:
        print(f"\nAn error occurred during the evaluation test: {e}")

    # 5. Test model aggregation
    print("\n--- Testing model aggregation ---")
    try:
        # Create a few dummy client model state_dicts
        weights1 = client_net.state_dict()
        weights2 = {key: val * 2 for key, val in weights1.items()} # Dummy weights
        
        aggregated = server.aggregate_models([weights1, weights2])
        
        # Check if the aggregated weights are the average
        assert torch.allclose(aggregated['features.0.weight'], (weights1['features.0.weight'] + weights2['features.0.weight']) / 2)
        print("Aggregation completed successfully.")

    except Exception as e:
        print(f"\nAn error occurred during the aggregation test: {e}")

    print("\nServer class test successful!")
