import os
import torch
from torch import nn
import torch.distributed as dist
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import dense_to_sparse
from torch.nn.attention import SDPBackend, sdpa_kernel
import time
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

from models.pi import AdjacencyGenerator
from models.GCN import GCN
from models.final_layer import FinalLayer
from helpers.data_loader import accuracy
from helpers.v import VNetwork
from helpers.sampling import sample_nodes
from helpers.weight_loader import load_all_weights, save_all_weights
from helpers.positional_encoding import positional_encoding
from helpers.config_loader import load_config
from helpers.visualize import visualize_tensor

# Load configuration
config = load_config()

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # Initialization parameters
    num_nodes = 2708
    num_model_layers = config['model']['num_model_layers']
    num_node_features = config['model']['num_node_features']
    num_classes = config['model']['num_classes']
    hidden_size = config['model']['hidden_size']
    d_model = config['model']['d_model']
    num_heads = config['model']['num_heads']
    d_ff = config['model']['d_ff']
    num_layers = config['model']['num_layers']
    dropout = config['model']['dropout']
    num_gcn_layers = config['model']['num_gcn_layers']
    epochs = config['training']['epochs']
    gamma = config['training']['gamma']
    pos_enc_dim = config['positional_encoding']['pos_enc_dim']
    device = rank

    # Load Cora dataset
    dataset = Planetoid(root='data/Cora', name='Cora')
    data = dataset[0].to(device)

    # Initialize NeighborLoader
    num_neighbors = [30] * 5

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=35,
        input_nodes=data.train_mask,
    )

    # Extract adjacency matrix, features, and labels
    adj = torch.zeros((num_nodes, num_nodes))
    adj[data.edge_index[0], data.edge_index[1]] = 1
    features = data.x
    labels = data.y
    idx_train = data.train_mask
    idx_val = data.val_mask
    idx_test = data.test_mask

    # Print shapes of loaded data
    if rank == 0:
        print(f"adj shape: {adj.shape}")
        print(f"features shape: {features.shape}")
        print(f"labels shape: {labels.shape}")
        print(f"idx_train shape: {idx_train.shape}")
        print(f"idx_val shape: {idx_val.shape}")
        print(f"idx_test shape: {idx_test.shape}")

    # Calculate positional encoding and add to features
    pos_enc = positional_encoding(adj, pos_enc_dim).to(device)
    features = torch.cat([features, pos_enc], dim=1)
    data.x = features
    # Print shapes after adding positional encoding
    if rank == 0:
        print(f"features shape after positional encoding: {features.shape}")

    # num_node_combined_featuresを設定
    num_node_combined_features = num_node_features + pos_enc_dim

    # Move labels to device
    labels = labels.to(device)
    features = features.to(device)
    adj = adj.to(device)  # adjもデバイスに移動

    # Initialize components
    adj_generators = [AdjacencyGenerator(d_model + pos_enc_dim, num_heads, d_ff, num_layers, hidden_size, device, dropout).to(device) for _ in range(num_model_layers)]
    gcn_models = [GCN(d_model + pos_enc_dim, hidden_size, num_node_combined_features, num_gcn_layers).to(device) for _ in range(num_model_layers)]
    final_layer = FinalLayer(num_node_combined_features, num_classes).to(device)  # FinalLayerの初期化
    v_networks = [VNetwork(d_model + pos_enc_dim, num_heads, d_ff, num_layers, 498, dropout).to(device) for _ in range(num_model_layers)]

    # To paralleliize for GPUs
    adj_generators = [DDP(adj_gen, device_ids=[rank], broadcast_buffers=False) for adj_gen in adj_generators]
    gcn_models = [DDP(gcn_model, device_ids=[rank], broadcast_buffers=False) for gcn_model in gcn_models]
    final_layer = DDP(final_layer, device_ids=[rank], broadcast_buffers=False)
    v_networks = [DDP(v_network, device_ids=[rank], broadcast_buffers=False) for v_network in v_networks]

    load_all_weights(adj_generators, gcn_models, v_networks, final_layer)

    # Set up optimizers
    optimizer_gcn = [optim.Adam(gcn_model.parameters(), lr=config['optimizer']['lr_gcn']) for gcn_model in gcn_models]
    optimizer_v = [optim.Adam(v_network.parameters(), lr=config['optimizer']['lr_v']) for v_network in v_networks]
    optimizer_adj = [optim.Adam(adj_generator.parameters(), lr=config['optimizer']['lr_adj'], maximize=True) for adj_generator in adj_generators]
    optimizer_final_layer = optim.Adam(final_layer.parameters(), lr=config['optimizer']['lr_final_layer'])

    # Create a file to log the epoch results
    log_file_path = 'training_log.txt'
    if rank == 0:
        with open(log_file_path, 'w') as f:
            f.write("Training Log\n")

    # Training loop
    for epoch in range(epochs):
        start_time = time.time()  # Start the timer at the beginning of the epoch
        epoch_acc = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for adj_generator in adj_generators:
            adj_generator.train()
        final_layer.train()
        for gcn_model in gcn_models:
            gcn_model.train()
        for v_network in v_networks:
            v_network.train()

        # バッチ処理のためのNeighborLoaderの反復処理
        for batch in train_loader:
            total_rewards = 0
            rewards_for_adj = []
            rewards_for_v = []
            log_probs_layers = []
            value_functions = []
            batch = batch.to(device)
            print(f"\nbatch: {batch}")
            updated_features = batch.x.clone()  # バッチ内の特徴量
            new_adj = torch.zeros((batch.num_nodes, batch.num_nodes), device=device)
            new_adj[batch.edge_index[0], batch.edge_index[1]] = 1
            adj_clone = new_adj.clone().detach()

            print(f"Batch size: {batch.batch_size}")  # バッチサイズ（シードノード数）
            print(f"Node IDs shape: {batch.n_id.shape}")  # 元のグラフにおけるノードIDのShape
            print(f"Edge index shape: {batch.edge_index.shape}")  # サブグラフのエッジインデックスのShape

            for layer in range(num_model_layers):
                print(f"\nLayer {layer + 1}/{num_model_layers}")

                # ノードをサンプリング
                sampled_indices = sample_nodes(updated_features, num_of_samples=498)
                sampled_indices_set = set(sampled_indices.tolist())  # サンプリングされたノードのセット
                
                # For each node, generate new neighbors using Bernoulli distribution
                layer_log_probs = []
                for node_idx in range(batch.num_nodes):
                    node_feature = updated_features[node_idx].unsqueeze(0)
                    neighbor_indices = adj_clone[node_idx].nonzero().view(-1)
                    neighbor_features = updated_features[neighbor_indices]

                # with sdpa_kernel(SDPBackend.MATH):  # adj_generatorにmathバックエンドを適用
                    adj_logits, new_neighbors = adj_generators[layer].module.generate_new_neighbors(node_feature, neighbor_features)

                    if node_idx in sampled_indices_set:
                        log_probs = nn.BCEWithLogitsLoss(reduction="sum")(adj_logits/3 + 1e-9, new_neighbors)
                        layer_log_probs.append(log_probs.item())

                    # Use the generated new neighbors to update the new adjacency matrix
                    for i, neighbor_idx in enumerate(neighbor_indices):
                        new_adj[node_idx, neighbor_idx] = new_neighbors[0, i, 0].item()

                print(f"adj_probs: {torch.sigmoid(adj_logits/3)}")
                print(f"new_neighbors: {new_neighbors}")

                log_probs_layers.append(sum(layer_log_probs) / len(layer_log_probs))
                
                # Sampled nodes for computing gradient and state value function V
                sampled_features = updated_features[sampled_indices].detach()
                print(f"Sampled features for layer {layer + 1}: {sampled_features.shape}")

                # Store log probabilities and value functions for later use
            # with sdpa_kernel(SDPBackend.MATH):  # VNetworkにmathバックエンドを適用
                value_function = v_networks[layer].module(sampled_features.unsqueeze(0)).view(-1)
                value_functions.append(value_function)
                print(f"Value function for layer {layer + 1}: {value_function}")

                # Forward pass through GCN using all nodes
                edge_index, edge_weight = dense_to_sparse(new_adj)
                node_features = gcn_models[layer].module(updated_features, edge_index)

                updated_features = node_features.clone()

                print(f"edge_index.shape: {edge_index.shape}")

                # Calculate reward
                sum_new_neighbors = new_adj.sum().item()  # 合計を計算
                print(f"sum_new_neighbors: {sum_new_neighbors}")
                log_sum = 1.0/torch.exp(torch.tensor(sum_new_neighbors /6000.0, device=device))  # sum_new_neighborsをtensorに変換
                reward = log_sum.item()

                rewards_for_adj.append(reward)
                rewards_for_v.append(reward)
                total_rewards += reward
                print(f"Reward for layer {layer + 1}: {reward}")

            output = final_layer.module(updated_features[:batch.batch_size])
            output_for_adj = final_layer.module(updated_features[:batch.batch_size])
            output = F.log_softmax(output, dim=1)
            print(f'output.shape: {output.shape}')
            
            acc = accuracy(output_for_adj, batch.y[:batch.batch_size])
            print(f"Training accuracy: {acc * 100:.2f}%")  # Print accuracy
            epoch_acc += acc
            # Calculate cumulative rewards for each layer
            cumulative_rewards = []
            for l in range(num_model_layers):
                cumulative_reward = sum(rewards_for_adj[l:]) + (num_model_layers * acc)
                cumulative_rewards.append(cumulative_reward)


            print(f"Cumulative rewards: {cumulative_rewards}")

            # Convert cumulative_rewards to FloatTensor
            cumulative_rewards = torch.tensor(cumulative_rewards, dtype=torch.float, device=device)

            # Calculate advantages
            advantages_layers = []
            for l in range(num_model_layers):
                advantages = cumulative_rewards[l] - value_functions[l]
                advantages_layers.append(advantages)
                print(f"Advantages for layer {l + 1}: {advantages.item()}")

            # Update GCN
            for opt_gcn in optimizer_gcn:
                opt_gcn.zero_grad()
            loss_gcn = F.nll_loss(output, batch.y[:batch.batch_size])
            print(f"GCN loss: {loss_gcn.item()}")
            loss_gcn.backward()
            for opt_gcn in optimizer_gcn:
                opt_gcn.step()

            print("init final_layer")
            # After all gradients are computed, step the optimizers
            optimizer_final_layer.zero_grad()
            optimizer_final_layer.step()

            count = 0
            # 各層の勾配計算とアドバンテージの適用
            for opt_adj, adj_generator in zip(optimizer_adj, adj_generators):
                # 各 optimizer_adj に対してゼロリセット
                opt_adj.zero_grad()

                log_probs_with_adv = log_probs_layers[count] * advantages_layers[count]

                # visualize_tensor(log_probs_with_adv, f"log_probs_with_adv_graph_{count}")
                
                log_probs_with_adv.backward(retain_graph=True)

                opt_adj.step()  # 各 optimizer_adj に対してステップ

                count = count + 1

            # Update V-networks
            for i, (v_network, v_opt) in enumerate(zip(v_networks, optimizer_v)):
                # print(f"i: {i}")
                v_opt.zero_grad()
                v_loss = F.mse_loss(value_functions[i], cumulative_rewards[i])
                # visualize_tensor(v_loss, output_path=f"v_loss_{i}")
                print(f"V-network loss for layer {i + 1}: {v_loss.item()}")
                v_loss.backward()
                v_opt.step()

        save_all_weights(adj_generators, gcn_models, v_networks, final_layer)

        end_time = time.time()
        epoch_time = end_time - start_time

        # Synchronize CUDA and wait for 2 seconds to ensure all operations are complete
        torch.cuda.synchronize()
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Epoch accuracy: {epoch_acc * 100:.2f}%")  # Print average accuracy across all batches
        print(f"Epoch time: {epoch_time}")
        # Write the results to the log file
        if rank == 0:
            with open(log_file_path, 'a') as f:
                f.write(f"\nEpoch {epoch + 1}/{epochs}\n")
                f.write(f"Epoch accuracy: {epoch_acc * 100:.2f}%\n")
                f.write(f"Epoch time: {epoch_time:.2f} seconds\n")  # Write the epoch time to the log file
                for i in range(num_model_layers):
                    f.write(f"Advantages for layer {i + 1}: {advantages_layers[i].item()}\n")

        time.sleep(2)

    print("Training finished and model weights saved!")

    # Test phase
    print("Starting testing phase...")

    for adj_generator in adj_generators:
        adj_generator.eval()
    final_layer.eval()
    for gcn_model in gcn_models:
        gcn_model.eval()
    for v_network in v_networks:
        v_network.eval()

    with torch.no_grad():
        for layer in range(num_model_layers):
            print(f"\nTesting Layer {layer + 1}/{num_model_layers}")

            new_adj = adj.clone()  # 新しい隣接行列を初期化

            for node_idx in range(num_nodes):
                node_feature = features[node_idx].unsqueeze(0)
                neighbor_indices = adj[node_idx].nonzero().view(-1).to('cpu')  # Get the indices of neighbors
                neighbor_features = features[neighbor_indices].to(device)
                
                with sdpa_kernel(SDPBackend.MATH):
                    adj_probs, new_neighbors = adj_generators[layer].module.generate_new_neighbors(node_feature, neighbor_features)

                for i, neighbor_idx in enumerate(neighbor_indices):
                    new_adj[node_idx, neighbor_idx] = new_neighbors[i]

            edge_index, edge_weight = dense_to_sparse(new_adj)
            data.edge_index = edge_index.to(device)
            data.edge_attr = edge_weight.to(device)
            data.x = features.to(device)  # テストデータもデバイスに移動
            node_features = gcn_models[layer].module(data.x, data.edge_index)

        output = final_layer.module(node_features[idx_test])
        output = F.log_softmax(output, dim=1)
        test_acc = accuracy(output, labels[idx_test])
        print(f"Test accuracy: {test_acc * 100:.2f}%")
        
        if rank == 0:
            with open(log_file_path, 'a') as f:
                f.write(f"\nTest accuracy: {test_acc * 100:.2f}%\n")

    print("Testing phase finished!")

    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size =  torch.cuda.device_count()
    run_demo(train, world_size)
