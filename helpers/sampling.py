import torch

def sample_nodes(features, num_of_samples):
    num_nodes = features.size(0)
    sample_size = int(num_of_samples)
    sampled_indices = torch.randperm(num_nodes)[:sample_size]
    return sampled_indices
