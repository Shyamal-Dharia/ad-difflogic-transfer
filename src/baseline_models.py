import numpy as np
import torch


N_CLASSES = 2
TARGET_PARAMETERS = 250_000
THERMOMETER_BINS = 15


def count_trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


class MLPBaseline(torch.nn.Module):
    def __init__(self, input_shape, hidden_dim, dropout=0.2):
        super().__init__()
        input_dim = int(np.prod(input_shape))
        self.layers = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, N_CLASSES),
        )

    def forward(self, x):
        return self.layers(x)


class Conv1dBaseline(torch.nn.Module):
    def __init__(self, input_shape, conv_channels, hidden_dim, dropout=0.2):
        super().__init__()
        input_dim = int(np.prod(input_shape))
        self.features = torch.nn.Sequential(
            torch.nn.Conv1d(1, conv_channels, kernel_size=5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Flatten(),
        )
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(conv_channels * input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, N_CLASSES),
        )

    def forward(self, x):
        x = x.flatten(start_dim=1).unsqueeze(1)
        x = self.features(x)
        return self.classifier(x)


class TransformerBaseline(torch.nn.Module):
    def __init__(self, input_shape, d_model, n_heads, feedforward_dim, dropout=0.2):
        super().__init__()
        self.n_tokens, self.token_features = transformer_token_shape(input_shape)
        self.class_token = torch.nn.Parameter(torch.zeros(1, 1, d_model))
        self.input_projection = torch.nn.Linear(self.token_features, d_model)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.head = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(d_model, N_CLASSES),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        else:
            x = x.reshape(x.shape[0], x.shape[1], -1)

        x = self.input_projection(x)
        class_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat([class_token, x], dim=1)
        x = self.encoder(x)
        return self.head(x[:, 0])


def transformer_token_shape(input_shape):
    if len(input_shape) == 1:
        return input_shape[0], 1

    return input_shape[0], int(np.prod(input_shape[1:]))


def closest_model(make_model, candidate_values, target_parameters):
    best_model = None
    best_distance = None

    for value in candidate_values:
        model = make_model(value)
        distance = abs(count_trainable_parameters(model) - target_parameters)

        if best_distance is None or distance < best_distance:
            best_model = model
            best_distance = distance

    return best_model


def make_mlp_250k_model(input_shape, dropout=0.2, target_parameters=TARGET_PARAMETERS):
    max_hidden = max(16, int(np.sqrt(target_parameters)) * 2)

    return closest_model(
        lambda hidden_dim: MLPBaseline(input_shape, hidden_dim, dropout=dropout),
        range(16, max_hidden + 1),
        target_parameters,
    )


def make_conv1d_250k_model(input_shape, dropout=0.2, target_parameters=TARGET_PARAMETERS):
    candidate_pairs = []
    for conv_channels in [8, 12, 16, 24, 32]:
        for hidden_dim in range(16, 513):
            candidate_pairs.append((conv_channels, hidden_dim))

    return closest_model(
        lambda values: Conv1dBaseline(
            input_shape,
            conv_channels=values[0],
            hidden_dim=values[1],
            dropout=dropout,
        ),
        candidate_pairs,
        target_parameters,
    )


def make_transformer_250k_model(input_shape, dropout=0.2, target_parameters=TARGET_PARAMETERS):
    candidate_values = []
    for d_model, n_heads in [(64, 4), (96, 4), (128, 4)]:
        for feedforward_dim in range(128, 1_537, 32):
            candidate_values.append((d_model, n_heads, feedforward_dim))

    return closest_model(
        lambda values: TransformerBaseline(
            input_shape,
            d_model=values[0],
            n_heads=values[1],
            feedforward_dim=values[2],
            dropout=dropout,
        ),
        candidate_values,
        target_parameters,
    )


def make_difflogic_medium_model(
    input_shape,
    thermometer_bins=THERMOMETER_BINS,
    target_parameters=TARGET_PARAMETERS,
    tau=30,
    device="cpu",
    implementation=None,
):
    try:
        from difflogic import GroupSum, LogicLayer
    except ModuleNotFoundError as exc:
        raise ImportError(
            "DiffLogic models require the difflogic package. "
            "Install requirements.txt before using model_kind='difflogic_medium'."
        ) from exc

    input_dim = int(np.prod(input_shape)) * thermometer_bins
    depth = 4
    width = max(1, round(target_parameters / (depth * 16)))
    layers = [torch.nn.Flatten()]
    layers.append(LogicLayer(input_dim, width, device=device, implementation=implementation))

    for _ in range(depth - 1):
        layers.append(LogicLayer(width, width, device=device, implementation=implementation))

    layers.append(GroupSum(k=N_CLASSES, tau=tau, device=device))
    return torch.nn.Sequential(*layers)


def make_baseline_model(
    model_kind,
    input_shape,
    dropout=0.2,
    target_parameters=TARGET_PARAMETERS,
    thermometer_bins=THERMOMETER_BINS,
    tau=30,
    device="cpu",
    implementation=None,
):
    if model_kind == "difflogic_medium":
        return make_difflogic_medium_model(
            input_shape,
            thermometer_bins=thermometer_bins,
            target_parameters=target_parameters,
            tau=tau,
            device=device,
            implementation=implementation,
        )
    if model_kind == "mlp_250k":
        return make_mlp_250k_model(input_shape, dropout=dropout, target_parameters=target_parameters)
    if model_kind == "conv1d_250k":
        return make_conv1d_250k_model(input_shape, dropout=dropout, target_parameters=target_parameters)
    if model_kind == "transformer_250k":
        return make_transformer_250k_model(input_shape, dropout=dropout, target_parameters=target_parameters)

    raise ValueError(f"Unknown baseline model kind: {model_kind}")
