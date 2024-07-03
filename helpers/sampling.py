import torch

def sample_nodes(batch_n_id, num_of_samples):
    num_nodes = batch_n_id.size(0)  # batch_n_idのノード数を取得
    sample_size = int(num_of_samples)  # サンプルサイズを整数に変換
    sampled_indices = torch.randperm(num_nodes)[:sample_size]  # ランダムなインデックスを生成
    return batch_n_id[sampled_indices]  # ランダムに選択されたノードIDを返す
