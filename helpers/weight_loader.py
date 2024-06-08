import os
import torch

# Create weights directory if it doesn't exist
weights_dir = 'weights'
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

def load_model_weights(model, filename):
    filepath = os.path.join(weights_dir, filename)
    if os.path.exists(filepath):
        try:
            model.load_state_dict(torch.load(filepath))
        except RuntimeError as e:
            print(f"Failed to load {filepath}: {e}")
            os.remove(filepath)  # Remove corrupted file
            raise FileNotFoundError(f"Corrupted file removed: '{filepath}'")
    else:
        raise FileNotFoundError(f"No such file: '{filepath}'")

def load_all_weights(adj_generators, gcn_models, v_networks, final_layer):
    def try_load(model, filename):
        try:
            load_model_weights(model, filename)
            print(f"Loaded {filename}")
        except FileNotFoundError as e:
            print(f"{e}")

    for i, adj_generator_model in enumerate(adj_generators):
        filename = f'adj_generator_{i}.pth'
        try_load(adj_generator_model, filename)
    for i, gcn_model in enumerate(gcn_models):
        filename = f'gcn_model_weights_{i}.pth'
        try_load(gcn_model, filename)
    for i, v_network in enumerate(v_networks):
        filename = f'v_network_weights_{i}.pth'
        try_load(v_network, filename)
    try_load(final_layer, 'final_layer_weights.pth')
    print("Model weights loading complete. Training will continue from the available weights.")

def save_model_weights(model, filename):
    filepath = os.path.join(weights_dir, filename)
    torch.save(model.state_dict(), filepath)

def save_all_weights(adj_generators, gcn_models, v_networks, final_layer):
    for i, adj_generator in enumerate(adj_generators):
        save_model_weights(adj_generator, f'adj_generator_{i}.pth')    
    for i, gcn_model in enumerate(gcn_models):
        save_model_weights(gcn_model, f'gcn_model_weights_{i}.pth')
    for i, v_network in enumerate(v_networks):
        save_model_weights(v_network, f'v_network_weights_{i}.pth')
    save_model_weights(final_layer, 'final_layer_weights.pth')
