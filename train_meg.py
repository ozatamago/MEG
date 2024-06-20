import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
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
import copy
import matplotlib.pyplot as plt

from models.pi import AdjacencyGenerator
from models.GCN import GCN
from models.final_layer import FinalLayer
from helpers.data_loader import accuracy
from helpers.v import VNetwork
from helpers.sampling import sample_nodes
from helpers.weight_loader import load_all_weights, save_all_weights, load_best_loss
from helpers.positional_encoding import positional_encoding
from helpers.config_loader import load_config
from helpers.visualize import visualize_tensor

# Load configuration
config = load_config()

# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

def print_memory_usage(device):
    print(f"Memory Allocated: {torch.cuda.memory_allocated(device) / (1024 * 1024):.2f} MB")
    print(f"Memory Cached: {torch.cuda.memory_reserved(device) / (1024 * 1024):.2f} MB")

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
checkpoint_path = os.path.join(checkpoint_dir, 'model.checkpoint')

# モデルの重みを保存する関数
def save_checkpoint(state, filename=checkpoint_path):
    torch.save(state, filename)

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
    num_neighbors = [5] * 5

    train_loader = NeighborLoader(
        data,
        num_neighbors=num_neighbors,
        batch_size=140,
        input_nodes=data.train_mask,
        shuffle=True,
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
    adj_generators = [AdjacencyGenerator(d_model + pos_enc_dim, num_heads, 8, device, dropout).to(device) for _ in range(num_model_layers)]
    gcn_models = [GCN(d_model + pos_enc_dim, hidden_size, num_node_combined_features, num_gcn_layers).to(device) for _ in range(num_model_layers)]
    final_layer = FinalLayer(num_node_combined_features, num_classes).to(device)  # FinalLayerの初期化
    v_networks = [VNetwork(d_model + pos_enc_dim, num_heads, d_ff, num_layers, 140, dropout).to(device) for _ in range(num_model_layers)]

    # To parallelize for GPUs
    adj_generators = [DDP(adj_gen, device_ids=[rank], broadcast_buffers=False) for adj_gen in adj_generators]
    gcn_models = [DDP(gcn_model, device_ids=[rank], broadcast_buffers=False) for gcn_model in gcn_models]
    final_layer = DDP(final_layer, device_ids=[rank], broadcast_buffers=False)
    v_networks = [DDP(v_network, device_ids=[rank], broadcast_buffers=False) for v_network in v_networks]

    # Ensure the weight files are present
    load_all_weights(adj_generators, gcn_models, v_networks, final_layer)

    best_acc = 0

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
    
    load_all_weights(adj_generators, gcn_models, v_networks, final_layer)
    best_loss = load_best_loss()

    # 配列を初期化
    epoch_acc_list = []
    val_acc_list = []
    val_loss_list = []
    
    # Training loop
    for epoch in range(epochs):
        dist.barrier()  # 各エポックの開始時に同期
        start_time = time.time()  # Start the timer at the beginning of the epoch
        epoch_acc = 0

        print(f"\nEpoch {epoch + 1}/{epochs}")
        for adj_generator in adj_generators:
            adj_generator.to(rank)
            adj_generator.train()
        final_layer.to(rank)
        final_layer.train()
        for gcn_model in gcn_models:
            gcn_model.to(rank)
            gcn_model.train()
        for v_network in v_networks:
            v_network.to(rank)
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

            for layer in range(num_model_layers):
                print(f"\nLayer {layer + 1}/{num_model_layers}")

                updated_features_for_adj = updated_features.clone().detach()

                # ノードをサンプリング
                sampled_indices = sample_nodes(updated_features, num_of_samples=140)

                adj_logits, new_neighbors = adj_generators[layer].module.generate_new_neighbors(batch.edge_index, updated_features_for_adj)
                
                # ログ確率の計算
                log_probs = nn.BCEWithLogitsLoss(reduction="sum")(adj_logits / 10 + 1e-9, new_neighbors.float())
                log_probs_layers.append(log_probs)
                print(f"log_probs_layers: {log_probs_layers}")

                # 新しい隣接行列を更新
                adj_clone = torch.zeros((batch.num_nodes, batch.num_nodes), device=device)
                adj_clone[batch.edge_index[0], batch.edge_index[1]] = new_neighbors.float()

                # Sampled nodes for computing gradient and state value function V
                sampled_features = updated_features[sampled_indices].detach()
                print(f"Sampled features for layer {layer + 1}: {sampled_features.shape}")

                value_function = v_networks[layer].module(sampled_features.unsqueeze(0)).view(-1)
                value_functions.append(value_function)
                print(f"Value function for layer {layer + 1}: {value_function}")

                # Forward pass through GCN using all nodes
                edge_index, _ = dense_to_sparse(adj_clone)
                node_features = gcn_models[layer].module(updated_features, edge_index)

                updated_features = node_features.clone()

                # Calculate reward
                sum_new_neighbors = adj_clone.sum().item()  # 合計を計算
                print(f"sum_new_neighbors: {sum_new_neighbors}")
                log_sum = 1.0 / torch.exp(torch.tensor(sum_new_neighbors / 2000.0, device=device))  # sum_new_neighborsをtensorに変換
                reward = log_sum.item()

                rewards_for_adj.append(reward)
                rewards_for_v.append(reward)
                total_rewards += reward
                print(f"Reward for layer {layer + 1}: {reward}")

            output = final_layer.module(updated_features[:batch.batch_size])
            output = F.log_softmax(output, dim=1)
            print(f'output.shape: {output.shape}')

            acc = accuracy(output, batch.y[:batch.batch_size])
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
            optimizer_final_layer.zero_grad()
            loss_gcn = F.nll_loss(output, batch.y[:batch.batch_size])
            print(f"GCN loss: {loss_gcn.item()}")
            # visualize_tensor(loss_gcn, f"gcn_loss_graph")
            loss_gcn.backward()
            for opt_gcn in optimizer_gcn:
                torch.nn.utils.clip_grad_norm_(gcn_model.parameters(), max_norm=0.1)
                opt_gcn.step()
            torch.nn.utils.clip_grad_norm_(final_layer.parameters(), max_norm=0.1)
            optimizer_final_layer.step()

            count = 0
            # 各層の勾配計算とアドバンテージの適用
            for opt_adj, adj_generator in zip(optimizer_adj, adj_generators):
                # print(count)
                # 各 optimizer_adj に対してゼロリセット
                opt_adj.zero_grad()
                log_probs_with_adv = log_probs_layers[count] * advantages_layers[count]
                # visualize_tensor(log_probs_with_adv, f"log_probs_with_adv_graph_{count}")
                # print(f"log_probs_with_adv: {log_probs_with_adv}")
                log_probs_with_adv.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(adj_generator.parameters(), max_norm=0.1)
                opt_adj.step()  # 各 optimizer_adj に対してステップ
                count = count + 1

            # Update V-networks
            for i, (v_network, v_opt) in enumerate(zip(v_networks, optimizer_v)):
                # print(f"i: {i}")
                v_opt.zero_grad()
                v_loss = F.mse_loss(value_functions[i], cumulative_rewards[i].unsqueeze(0))
                # visualize_tensor(v_loss, output_path=f"v_loss_{i}")
                print(f"V-network loss for layer {i + 1}: {v_loss.item()}")
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(v_network.parameters(), max_norm=0.1)
                v_opt.step()

            print("gradient computation is finished!")

        # save_all_weights(adj_generators, gcn_models, v_networks, final_layer)

        # Synchronize CUDA and wait for 2 seconds to ensure all operations are complete
        torch.cuda.synchronize()
        dist.all_reduce(epoch_acc, op=dist.ReduceOp.SUM)
        epoch_acc /= world_size

        if rank == 0:
            print("validation start")
            for adj_generator in adj_generators:
                adj_generator.to(rank)
                adj_generator.eval()
            final_layer.to(rank)
            final_layer.eval()
            for gcn_model in gcn_models:
                gcn_model.to(rank)
                gcn_model.eval()

            with torch.no_grad():
                val_loss = 0
                val_acc = 0
                print("validation computation start")
                new_adj_for_val = adj.clone()
                node_features_for_val = features.clone()
                edge_index = data.edge_index.clone()

                for layer in range(num_model_layers):
                    print(f"validation layer: {layer}")

                    # 全ノードに対して新しい隣接行列を一括で生成
                    _, new_neighbors_for_val = adj_generators[layer].module.generate_new_neighbors(edge_index, node_features_for_val)
                    
                    # 新しい隣接行列を更新
                    new_adj_for_val = torch.zeros((num_nodes, num_nodes), device=device)
                    new_adj_for_val[edge_index[0], edge_index[1]] = new_neighbors_for_val.float()

                    print(f"new_adj.sum: {new_adj_for_val.sum().item()}")

                    # GCNレイヤーのフォワードパスを通して特徴量を更新
                    edge_index_for_val, _ = dense_to_sparse(new_adj_for_val)
                    node_features_for_val = gcn_models[layer].module(node_features_for_val, edge_index_for_val)

                val_output = final_layer.module(node_features_for_val[idx_val])
                val_output = F.log_softmax(val_output, dim=1)
                val_loss = F.nll_loss(val_output, labels[idx_val])
                val_acc = accuracy(val_output, labels[idx_val])
                print(f"Validation Loss: {val_loss.item()}, Validation Accuracy: {val_acc.item()}")


             # or val_acc.item() > best_acc
            
            # バリデーション損失が改善された場合、または精度が向上した場合にモデルを保存
            if val_loss.item() < best_loss:
                print("best_loss is updated!")
                best_loss = val_loss.item()
                best_acc = val_acc
                save_all_weights(adj_generators, gcn_models, v_networks, final_layer, best_loss)
                # save_checkpoint({
                #     'epoch': epoch,
                #     'state_dict': {
                #         'adj_generators': [adj_generator.module.state_dict() for adj_generator in adj_generators],
                #         'gcn_models': [gcn_model.module.state_dict() for gcn_model in gcn_models],
                #         'v_networks': [v_network.module.state_dict() for v_network in v_networks],
                #         'final_layer': final_layer.module.state_dict()
                #     },
                #     'optimizer': {
                #         'optimizer_adj': [opt.state_dict() for opt in optimizer_adj],
                #         'optimizer_gcn': [opt.state_dict() for opt in optimizer_gcn],
                #         'optimizer_v': [opt.state_dict() for opt in optimizer_v],
                #         'optimizer_final_layer': optimizer_final_layer.state_dict()
                #     },
                #     'best_loss': best_loss
                # })       
            else:
                bad_counter+=1
            
            end_time = time.time()
            epoch_time = end_time - start_time

            # 配列に追加
            epoch_acc_list.append(epoch_acc.item())
            val_acc_list.append(val_acc.item())
            val_loss_list.append(val_loss.item())
            
            print(f"Epoch {epoch + 1}/{epochs}")
            print(f"Epoch accuracy: {epoch_acc * 100:.2f}%")
            print(f"Validation accuracy: {val_acc.item()}")
            print(f"best accuracy: {best_acc}")
            print(f"Validation loss: {val_loss.item()}")
            print(f"best loss: {best_loss}")
            print(f"Epoch time: {epoch_time}")
        
            with open(log_file_path, 'a') as f:
                f.write(f"\nEpoch {epoch + 1}/{epochs}\n")
                f.write(f"Epoch accuracy: {epoch_acc * 100:.2f}%\n")
                f.write(f"Validation accuracy: {val_acc.item()}\n")
                f.write(f"Best  accuracy: {best_acc}\n")
                f.write(f"Validation loss: {val_loss.item()}\n")
                f.write(f"Best loss: {best_loss}\n")
                f.write(f"Epoch time: {epoch_time:.2f} seconds\n")
                if bad_counter == patience:
                    f.write("Optimization Finished!")
                    break
    
    print("Training finished and model weights saved!")

    load_all_weights(adj_generators, gcn_models, v_networks, final_layer)

    
    # # プロットする関数を定義
    # def plot_metrics(epoch_acc_list, val_acc_list, val_loss_list, output_dir='plots'):
    #     if not os.path.exists(output_dir):
    #         os.makedirs(output_dir)
        
    #     plt.figure(figsize=(12, 5))
    
    #     # ACCのプロット
    #     plt.subplot(1, 2, 1)
    #     plt.plot(epoch_acc_list, label='Training Accuracy')
    #     plt.plot(val_acc_list, label='Validation Accuracy')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Accuracy')
    #     plt.title('Training and Validation Accuracy')
    #     plt.legend()
    
    #     # Lossのプロット
    #     plt.subplot(1, 2, 2)
    #     plt.plot(val_loss_list, label='Validation Loss')
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.title('Validation Loss')
    #     plt.legend()
    
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, 'training_validation_metrics.png'))
    #     plt.close()

    # # プロットを呼び出す
    # if rank == 0:
    #     plot_metrics(epoch_acc_list, val_acc_list, val_loss_list)

    # Test phase
    print("Starting testing phase...")

    # Load the best checkpoint
    # best_checkpoint = torch.load(checkpoint_path, map_location={'cuda:0': 'cuda:%d' % rank})
    # for i, adj_generator in enumerate(adj_generators):
    #     adj_generator.module.load_state_dict(best_checkpoint['state_dict']['adj_generators'][i])
    # for i, gcn_model in enumerate(gcn_models):
    #     gcn_model.module.load_state_dict(best_checkpoint['state_dict']['gcn_models'][i])
    # for i, v_network in enumerate(v_networks):
    #     v_network.module.load_state_dict(best_checkpoint['state_dict']['v_networks'][i])
    # final_layer.module.load_state_dict(best_checkpoint['state_dict']['final_layer'])
    
    for adj_generator in adj_generators:
        adj_generator.eval()
    final_layer.eval()
    for gcn_model in gcn_models:
        gcn_model.eval()
    for v_network in v_networks:
        v_network.eval()

    with torch.no_grad():
        node_features = features.clone()
        new_adj = adj.clone()  # 新しい隣接行列を初期化
        edge_index = data.edge_index.clone()
        for layer in range(num_model_layers):
            print(f"\nTesting Layer {layer + 1}/{num_model_layers}")

            # 全ノードに対して新しい隣接行列を一括で生成
            _, new_neighbors = adj_generators[layer].module.generate_new_neighbors(edge_index, node_features)

            # 新しい隣接行列を更新
            new_adj = torch.zeros((num_nodes, num_nodes), device=device)
            new_adj[edge_index[0], edge_index[1]] = new_neighbors.float()

            print(f"new_adj.sum: {new_adj.sum().item()}")
            new_edge_index, _ = dense_to_sparse(new_adj)
            node_features = gcn_models[layer].module(node_features, new_edge_index)

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
