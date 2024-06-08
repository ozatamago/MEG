import torch
from torchviz import make_dot

def visualize_tensor(tensor, output_path="tensor_graph"):
    """
    Visualize the computation graph of a given tensor.
    
    Parameters:
    - tensor: The tensor to visualize.
    - output_path: Path to save the output graph visualization.
    """
    graph = make_dot(tensor, params={"tensor": tensor})
    graph.render(output_path, format="png")
    print(f"Graph saved to {output_path}.png")
