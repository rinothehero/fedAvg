import torch
import torch.nn as nn
import torch.optim as optim
from models.vgg import ClientVGG

class Client:
    """
    Represents a client in the Split Federated Learning setup.
    """
    def __init__(self, client_id, train_loader, client_model, device, learning_rate=0.01, mu=0.0):
        """
        Initializes the Client object.

        Args:
            client_id (int): A unique identifier for the client.
            train_loader (DataLoader): The DataLoader for the client's local training data.
            client_model (ClientVGG): The client-side part of the VGG model.
            device (torch.device): The device to run the model on (e.g., 'cuda:0' or 'cpu').
            learning_rate (float): The learning rate for the optimizer.
            mu (float): The proximal term coefficient for FedProx.
        """
        self.client_id = client_id
        self.train_loader = train_loader
        self.device = device
        self.model = client_model
        self.optimizer = optim.SGD(self.model.parameters(), lr=learning_rate)
        self.mu = mu

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

    def train(self, server, global_model, local_epochs):
        """
        Performs the local training loop for a specified number of epochs.
        Includes the FedProx proximal term if mu > 0.

        Args:
            server (Server): The central server instance.
            global_model (nn.Module): The global client model from the start of the round (w^t).
            local_epochs (int): The number of local epochs to train for.

        Returns:
            list: A list of loss values for each training step.
        """
        losses = []
        self.model.train()
        
        # Get the parameters of the global model for the proximal term
        global_params = list(global_model.parameters())

        for epoch in range(local_epochs):
            for data, labels in self.train_loader:
                # 1. Client-side forward pass
                smashed_data = self.forward(data)
                
                # 2. Server computes its forward/backward pass and returns gradients
                smashed_data_grad, loss = server.train_step(smashed_data, labels)
                losses.append(loss)
                
                # 3. Client-side backward pass
                self.optimizer.zero_grad()
                smashed_data.backward(gradient=smashed_data_grad.to(self.device))
                
                # 4. Add FedProx proximal term (if mu > 0)
                if self.mu > 0:
                    local_params = list(self.model.parameters())
                    for local_param, global_param in zip(local_params, global_params):
                        # The proximal term is 0.5 * mu * ||w - w^t||^2
                        # Its gradient is mu * (w - w^t)
                        prox_term_grad = self.mu * (local_param - global_param.detach())
                        if local_param.grad is not None:
                            local_param.grad += prox_term_grad
                        # In some cases, a parameter might not have a gradient (e.g., if it's frozen)
                        # We don't need to handle that case here as VGG has no such layers.
                
                # 5. Update client model
                self.optimizer.step()
                
        return losses

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

    # --- Mock Server for testing client.train() ---
    class MockServer:
        """A mock server that simulates the server's train_step."""
        def train_step(self, smashed_data, labels):
            # Return dummy gradient and loss
            return torch.randn_like(smashed_data), 0.5

    print("--- Initializing a test client ---")
    
    # 1. Prepare dummy data and model
    train_loaders, _, _ = get_cifar10_dataloaders(num_clients=1, batch_size=256)
    test_client_loader = train_loaders[0]
    
    client_net, _ = split_vgg11_for_cifar10()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    client_net.to(device)

    # 2. Create a Client instance
    client = Client(
        client_id=0, 
        train_loader=test_client_loader, 
        client_model=client_net, 
        device=device,
        mu=0.1  # Use a non-zero mu to test FedProx path
    )
    print(f"Client {client.client_id} initialized on device: {client.device}")

    # 3. Simulate local training using the new train method
    print("\n--- Simulating local training ---")
    try:
        # a. Create a mock server and a dummy global model
        mock_server = MockServer()
        global_model, _ = split_vgg11_for_cifar10()
        global_model.to(device)

        # b. Run the train method for one epoch
        losses = client.train(server=mock_server, global_model=global_model, local_epochs=1)
        print(f"Client train method completed successfully. Ran for {len(losses)} steps.")
        
        print("\nClient class test successful!")

    except Exception as e:
        print(f"\nAn error occurred during the test: {e}")
        raise e

