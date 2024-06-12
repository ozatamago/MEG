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

def save_all_weights(adj_generators, gcn_models, v_networks, final_layer, best_loss):
    for i, adj_generator in enumerate(adj_generators):
        save_model_weights(adj_generator, f'adj_generator_{i}.pth')    
    for i, gcn_model in enumerate(gcn_models):
        save_model_weights(gcn_model, f'gcn_model_weights_{i}.pth')
    for i, v_network in enumerate(v_networks):
        save_model_weights(v_network, f'v_network_weights_{i}.pth')
    save_model_weights(final_layer, 'final_layer_weights.pth')
    best_loss_filepath = os.path.join(weights_dir, 'best_loss.txt')
    with open(best_loss_filepath, 'w') as f:
        f.write(str(best_loss.item() if isinstance(best_loss, torch.Tensor) else best_loss))

def load_best_loss(directory='weights'):
    best_loss_filepath = os.path.join(directory, 'best_loss.txt')
    if os.path.exists(best_loss_filepath):
        with open(best_loss_filepath, 'r') as f:
            best_loss_str = f.read().strip()
            try:
                best_loss = float(best_loss_str)
            except ValueError:
                # テンソル形式の文字列を処理
                if best_loss_str.startswith("tensor"):
                    best_loss = eval(best_loss_str).item()
                else:
                    raise ValueError(f"Unexpected format in best_loss file: {best_loss_str}")
        print(f"Loaded best_loss: {best_loss}")
    else:
        best_loss = 1000.0
        print(f"No best_loss file found. Using default value: {best_loss}")
    return best_loss
