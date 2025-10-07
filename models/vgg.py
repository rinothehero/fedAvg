import torch
import torch.nn as nn
import torchvision.models as models

class ClientVGG(nn.Module):
    """Client-side part of the VGG-11 model."""
    def __init__(self, client_features):
        super(ClientVGG, self).__init__()
        self.features = client_features

    def forward(self, x):
        return self.features(x)

class ServerVGG(nn.Module):
    """Server-side part of the VGG-11 model, adapted for CIFAR-10."""
    def __init__(self, server_features, classifier):
        super(ServerVGG, self).__init__()
        self.features = server_features
        self.classifier = classifier

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def split_vgg11_for_cifar10(split_point=5):
    """
    Splits the VGG-11 model, adapted for CIFAR-10, into a client-side and a server-side model.
    A split_point of 5 splits after the second Conv layer's ReLU.
    """
    vgg11 = models.vgg11(weights=None)

    # Adapt VGG-11 for CIFAR-10
    features_list = list(vgg11.features.children())[:-1] 
    classifier = nn.Sequential(
        nn.Linear(512 * 2 * 2, 4096),
        nn.ReLU(True), nn.Dropout(),
        nn.Linear(4096, 4096),
        nn.ReLU(True), nn.Dropout(),
        nn.Linear(4096, 10),
    )

    if not 0 < split_point < len(features_list):
        raise ValueError(f"Split point must be between 1 and {len(features_list)-1}")

    client_features = nn.Sequential(*features_list[:split_point])
    server_features = nn.Sequential(*features_list[split_point:])
    
    client_model = ClientVGG(client_features)
    server_model = ServerVGG(server_features, classifier)
    
    return client_model, server_model

if __name__ == '__main__':
    client_net, server_net = split_vgg11_for_cifar10()
    
    print("--- Client Model Architecture ---")
    print(client_net)
    
    print("\n--- Server Model Architecture ---")
    print(server_net)
    
    print("\n--- Testing Forward Pass ---")
    dummy_input = torch.randn(256, 3, 32, 32) 
    print(f"Initial input shape: {dummy_input.shape}")
    
    smashed_data = client_net(dummy_input)
    print(f"Output shape from client (smashed_data): {smashed_data.shape}")
    
    final_output = server_net(smashed_data)
    print(f"Final output shape from server: {final_output.shape}")
    
    assert final_output.shape == (256, 10)
    print(f"\nTest successful: Final output shape is correct.")