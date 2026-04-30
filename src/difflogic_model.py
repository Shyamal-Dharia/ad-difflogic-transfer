import torch
import numpy as np
from difflogic import GroupSum, LogicLayer


INPUT_DIM = 1125
N_CLASSES = 2

MODEL_SIZES = {
    "small": {
        "width": 800,
        "depth": 4,
    },
    "medium": {
        "width": 1_600,
        "depth": 4,
    },
    "large": {
        "width": 4_000,
        "depth": 4,
    },
}


def count_logic_gates(width, depth):
    return width * depth


def count_trainable_logic_logits(width, depth):
    return count_logic_gates(width, depth) * 16


def make_difflogic_model(
    size="small",
    tau=30,
    device="cuda",
    input_dim=INPUT_DIM,
    implementation=None,
):
    config = MODEL_SIZES[size]
    width = config["width"]
    depth = config["depth"]

    layers = [torch.nn.Flatten()]
    layers.append(LogicLayer(input_dim, width, device=device, implementation=implementation))

    for _ in range(depth - 1):
        layers.append(LogicLayer(width, width, device=device, implementation=implementation))

    layers.append(GroupSum(k=N_CLASSES, tau=tau, device=device))
    return torch.nn.Sequential(*layers)


def logic_layers(model):
    return [layer for layer in model if isinstance(layer, LogicLayer)]


def get_logic_connections(model):
    return [
        (
            layer.indices[0].detach().cpu(),
            layer.indices[1].detach().cpu(),
        )
        for layer in logic_layers(model)
    ]


def set_logic_connections(model, connections, device):
    for layer, indices in zip(logic_layers(model), connections):
        index_a, index_b = indices
        layer.indices = (
            index_a.to(device=device, dtype=torch.int64).contiguous(),
            index_b.to(device=device, dtype=torch.int64).contiguous(),
        )

        if layer.implementation == "cuda":
            rebuild_cuda_connection_index(layer, device)


def rebuild_cuda_connection_index(layer, device):
    given_x_indices_of_y = [[] for _ in range(layer.in_dim)]
    indices_0 = layer.indices[0].detach().cpu().numpy()
    indices_1 = layer.indices[1].detach().cpu().numpy()

    for output_index in range(layer.out_dim):
        given_x_indices_of_y[indices_0[output_index]].append(output_index)
        given_x_indices_of_y[indices_1[output_index]].append(output_index)

    layer.given_x_indices_of_y_start = torch.tensor(
        np.array([0] + [len(indices) for indices in given_x_indices_of_y]).cumsum(),
        device=device,
        dtype=torch.int64,
    )
    layer.given_x_indices_of_y = torch.tensor(
        [item for indices in given_x_indices_of_y for item in indices],
        dtype=torch.int64,
        device=device,
    )


def print_model_sizes():
    for size, config in MODEL_SIZES.items():
        width = config["width"]
        depth = config["depth"]
        gates = count_logic_gates(width, depth)
        logits = count_trainable_logic_logits(width, depth)
        print(
            f"{size}: width={width}, depth={depth}, "
            f"logic_gates={gates:,}, trainable_logic_logits={logits:,}"
        )


if __name__ == "__main__":
    print_model_sizes()
