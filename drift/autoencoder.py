"""PyTorch autoencoder producing a multivariate drift signal via reconstruction error."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class DriftAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, bottleneck_dim: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def train_autoencoder(
    X: np.ndarray, epochs: int = 30, lr: float = 1e-3, batch_size: int = 256
) -> DriftAutoencoder:
    """Train on standardized reference feature vectors — the same fitted ColumnTransformer
    output the Phase 1 champion model consumes (model.features.build_preprocessor). Fit
    only on the reference window; never on a comparison window."""
    torch.manual_seed(42)
    model = DriftAutoencoder(input_dim=X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed = model(batch)
            loss = loss_fn(reconstructed, batch)
            loss.backward()
            optimizer.step()

    return model


def reconstruction_error(model: DriftAutoencoder, X: np.ndarray) -> np.ndarray:
    """Per-row mean squared reconstruction error."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        reconstructed = model(X_tensor)
        errors = ((reconstructed - X_tensor) ** 2).mean(dim=1)
    return errors.numpy()
