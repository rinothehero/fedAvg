import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import ClientVGG

class Client:
    """
    Represents a client in the Split Federated Learning setup.
    """
    def __init__(self, client_id, train_loader, client_model, device, learning_rate=0.01):
        """
        Initializes the Client object.

        Args:
            client_id (int): A unique identifier for the client.
            train_loader (DataLoader): The DataLoader for the client's local training data.
            client_model (ClientVGG): The client-side part of the VGG model.
            device (torch.device): The device to run the model on (e.g., 'cuda:0' or 'cpu').
            learning_rate (float): The learning rate for the optimizer.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = client_model
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)

    def forward(self, data):
        """
        Performs the forward pass on the client's model.

        Args:
            data (torch.Tensor): The input data (a batch of images).

        Returns:
            torch.Tensor: The output from the client model (smashed_data),
                          which will be sent to the server.
        """
        # Ensure data is on the correct device
        data = data.to(self.device)
        
        # Perform the forward pass
        smashed_data = self.model(data)
        return smashed_data

    def backward(self, smashed_data, smashed_data_grad):
        """
        Performs the backward pass and updates the client's model.

        Args:
            smashed_data (torch.Tensor): The output from the client's forward pass.
            smashed_data_grad (torch.Tensor): The gradients for the smashed_data,
                                              received from the server.
        """
        # Clear previous gradients
        self.optimizer.zero_grad()
        
        # Perform the backward pass starting from the smashed_data
        smashed_data.backward(gradient=smashed_data_grad.to(self.device))
        
        # Update the model's parameters
        self.optimizer.step()

    def get_model_weights(self):
        """Returns the state dictionary of the client's model."""
        return self.model.state_dict()

    def set_model_weights(self, state_dict):
        """Sets the state dictionary of the client's model."""
        self.model.load_state_dict(state_dict)


if __name__ == '__main__':
    # --- Test the Client class functionality ---
    from models.vgg import split_vgg11_for_cifar10
    from utils.data_loader import get_cifar10_dataloaders

    print("--- Initializing a test client ---")
    
    # 1. Prepare dummy data and model
    # Use the data loader to get a sample batch
    train_loaders, _, _ = get_cifar10_dataloaders(num_clients=1, batch_size=256)
    print(f"Train loader for client 0: {train_loaders[0]}")
    test_client_loader = train_loaders[0]
    print(f"Number of batches in client 0's train loader: {len(test_client_loader)}")
    
    # Create a client model instance
    client_net, _ = split_vgg11_for_cifar10()

    # 2. Create a Client instance
    client = Client(client_id=0, train_loader=test_client_loader, client_model=client_net)
    print(f"Client {client.client_id} initialized on device: {client.device}")

    # 3. Simulate a single training step
    print("\n--- Simulating a single training step ---")
    try:
        # Get a batch of data
        data, _ = next(iter(client.train_loader))
        print(f"Input data shape: {data.shape}")

        # a. Client performs forward pass
        smashed_data = client.forward(data)
        print(f"Smashed data shape: {smashed_data.shape}")
        
        # b. Simulate server processing and gradient return
        # In a real scenario, the server would compute this gradient.
        # Here, we create a dummy gradient for testing purposes.
        dummy_smashed_data_grad = torch.randn_like(smashed_data)
        print(f"Dummy gradient shape from server: {dummy_smashed_data_grad.shape}")

        # c. Client performs backward pass with the received gradient
        client.backward(smashed_data, dummy_smashed_data_grad)
        print("Client backward pass and optimizer step completed successfully.")
        
        print("\nClient class test successful!")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")

