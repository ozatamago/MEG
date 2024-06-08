import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# データセットの読み込み
dataset = Planetoid(root='data/Cora', name='Cora', transform=T.NormalizeFeatures())
data = dataset[0].to(device)

# データセット情報の表示
print(f"Dataset: {data}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of features: {dataset.num_node_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Features: {data.x}")

# 8ホップ先までのNeighborLoaderの初期化
num_hops = 8
num_neighbors = [30] * num_hops  # 各ホップでサンプリングする近隣ノード数を30に設定

train_loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    batch_size=128,
    input_nodes=data.train_mask,
)

test_loader = NeighborLoader(
    data,
    num_neighbors=num_neighbors,
    batch_size=128,
    input_nodes=data.test_mask,
)

# トレーニングのためのNeighborLoaderの反復処理
for batch in train_loader:
    batch = batch.to(device)
    x = batch.x  # バッチ内のノード特徴量
    y = batch.y[:batch.batch_size]  # バッチ内のラベル（シードノードのラベル）

    print("=====================================")
    print(f"Batch size: {batch.batch_size}")  # バッチサイズ（シードノード数）
    print(f"Node IDs shape: {batch.n_id.shape}")  # 元のグラフにおけるノードIDのShape
    print(f"Edge index shape: {batch.edge_index.shape}")  # サブグラフのエッジインデックスのShape
    print(f"Batch features shape: {x.shape}")  # ノードの特徴量のShape
    print(f"Batch labels shape: {y.shape}")  # ノードのラベルのShape

# テストのためのNeighborLoaderの反復処理
for batch in test_loader:
    batch = batch.to(device)
    x = batch.x  # バッチ内のノード特徴量
    y = batch.y[:batch.batch_size]  # バッチ内のラベル（シードノードのラベル）

    print("=====================================")
    print(f"Batch size: {batch.batch_size}")  # バッチサイズ（シードノード数）
    print(f"Node IDs shape: {batch.n_id.shape}")  # 元のグラフにおけるノードIDのShape
    print(f"Edge index shape: {batch.edge_index.shape}")  # サブグラフのエッジインデックスのShape
    print(f"Batch features shape: {x.shape}")  # ノードの特徴量のShape
    print(f"Batch labels shape: {y.shape}")  # ノードのラベルのShape
