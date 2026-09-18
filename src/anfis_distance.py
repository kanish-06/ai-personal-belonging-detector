"""
anfis_distance.py — ANFIS Distance Estimation Module

Implements an Adaptive Neuro-Fuzzy Inference System (ANFIS) that maps
bounding box area ratio → estimated real-world distance in centimeters.

ANFIS architecture (single-input, single-output Sugeno-type):
  Layer 1: Fuzzification — Gaussian membership functions on box_area_ratio
  Layer 2: Rule firing strengths (one per MF)
  Layer 3: Normalization of firing strengths
  Layer 4: Consequent — first-order Sugeno: f_i = p_i * x + q_i
  Layer 5: Defuzzification — weighted average of consequent outputs

The entire network is differentiable and trained end-to-end with gradient descent
on (box_area_ratio, true_distance_cm) calibration pairs.
"""

import os
import pickle
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[WARNING] PyTorch not installed. ANFIS training will not work.")
    print("         Install with: pip install torch")


# ─── ANFIS Model (PyTorch) ─────────────────────────────────────────────────────

if HAS_TORCH:
    class ANFISModel(nn.Module):
        """
        Single-input, single-output ANFIS (Sugeno first-order).
        
        Architecture:
            - n_mfs Gaussian membership functions on the input
            - n_mfs first-order Sugeno consequents: y_i = p_i * x + q_i
            - Output = weighted average of consequents by normalized firing strengths
        """

        def __init__(self, n_mfs=5, input_min=0.0, input_max=0.5):
            """
            Args:
                n_mfs (int): Number of membership functions / fuzzy rules.
                input_min (float): Expected minimum of box_area_ratio.
                input_max (float): Expected maximum of box_area_ratio.
            """
            super().__init__()
            self.n_mfs = n_mfs

            # Layer 1: Gaussian MF parameters (centers and widths)
            # Initialize centers evenly spaced across input range
            centers = torch.linspace(input_min, input_max, n_mfs)
            self.centers = nn.Parameter(centers)

            # Initialize widths based on spacing
            spread = (input_max - input_min) / (n_mfs - 1) if n_mfs > 1 else 0.1
            widths = torch.full((n_mfs,), spread * 0.6)
            self.widths = nn.Parameter(widths)

            # Layer 4: Sugeno consequent parameters (first-order: p*x + q)
            self.p = nn.Parameter(torch.randn(n_mfs) * 0.1)  # slopes
            self.q = nn.Parameter(torch.randn(n_mfs) * 0.1)  # intercepts

        def forward(self, x):
            """
            Forward pass.
            
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, 1) — box_area_ratio values.
                
            Returns:
                torch.Tensor: Predicted distances of shape (batch_size, 1).
            """
            # x shape: (batch, 1)
            # Layer 1: Fuzzification — Gaussian membership degrees
            # mu_i(x) = exp(-(x - c_i)^2 / (2 * sigma_i^2))
            # Ensure widths are positive
            widths = torch.abs(self.widths) + 1e-8
            # (batch, 1) - (n_mfs,) → (batch, n_mfs)
            mu = torch.exp(-((x - self.centers) ** 2) / (2 * widths ** 2))

            # Layer 2: Rule firing strengths (same as mu for single-input case)
            w = mu  # (batch, n_mfs)

            # Layer 3: Normalized firing strengths
            w_sum = w.sum(dim=1, keepdim=True) + 1e-8
            w_norm = w / w_sum  # (batch, n_mfs)

            # Layer 4: First-order Sugeno consequents
            # f_i = p_i * x + q_i
            f = self.p * x + self.q  # (batch, n_mfs)

            # Layer 5: Defuzzification — weighted average
            y = (w_norm * f).sum(dim=1, keepdim=True)  # (batch, 1)

            return y

        def get_mf_params(self):
            """Return membership function parameters for inspection."""
            return {
                'centers': self.centers.detach().cpu().numpy(),
                'widths': torch.abs(self.widths).detach().cpu().numpy(),
                'p': self.p.detach().cpu().numpy(),
                'q': self.q.detach().cpu().numpy(),
            }


# ─── Training and Inference Wrapper ────────────────────────────────────────────

class ANFISDistanceEstimator:
    """
    High-level wrapper for training, saving, loading, and using the ANFIS
    distance estimation model.
    """

    DEFAULT_MODEL_PATH = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "models", "anfis_distance_model.pkl"
    )

    def __init__(self, n_mfs=5, model_path=None):
        """
        Args:
            n_mfs (int): Number of fuzzy membership functions.
            model_path (str): Path to save/load the trained model.
        """
        self.n_mfs = n_mfs
        self.model_path = model_path or self.DEFAULT_MODEL_PATH
        self.model = None
        self._is_trained = False

        # Normalization parameters (set during training)
        self._input_min = 0.0
        self._input_max = 0.5
        self._output_min = 20.0
        self._output_max = 300.0

    def train(self, box_area_ratios, true_distances_cm, 
              epochs=2000, lr=0.01, verbose=True):
        """
        Train the ANFIS model on calibration data.

        Args:
            box_area_ratios (array-like): Input values (box area / frame area).
            true_distances_cm (array-like): Ground truth distances in centimeters.
            epochs (int): Number of training epochs.
            lr (float): Learning rate.
            verbose (bool): Print training progress.

        Returns:
            dict: Training history with loss values.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for ANFIS training.")

        X = np.array(box_area_ratios, dtype=np.float32).reshape(-1, 1)
        Y = np.array(true_distances_cm, dtype=np.float32).reshape(-1, 1)

        # Store normalization parameters
        self._input_min = float(X.min())
        self._input_max = float(X.max())
        self._output_min = float(Y.min())
        self._output_max = float(Y.max())

        # Convert to tensors
        X_tensor = torch.FloatTensor(X)
        Y_tensor = torch.FloatTensor(Y)

        # Create model
        self.model = ANFISModel(
            n_mfs=self.n_mfs,
            input_min=self._input_min,
            input_max=self._input_max,
        )

        # Training setup
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=200, factor=0.5, min_lr=1e-6
        )
        loss_fn = nn.MSELoss()

        history = {'loss': [], 'mae': []}
        best_loss = float('inf')
        best_state = None

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            y_pred = self.model(X_tensor)
            loss = loss_fn(y_pred, Y_tensor)
            mae = torch.mean(torch.abs(y_pred - Y_tensor)).item()

            loss.backward()
            optimizer.step()
            scheduler.step(loss.item())

            history['loss'].append(loss.item())
            history['mae'].append(mae)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}

            if verbose and (epoch + 1) % 200 == 0:
                print(f"Epoch {epoch+1}/{epochs} — Loss: {loss.item():.4f}, MAE: {mae:.2f} cm")

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self._is_trained = True

        if verbose:
            print(f"\nTraining complete. Best MSE: {best_loss:.4f}")
            print(f"MF Parameters: {self.model.get_mf_params()}")

        return history

    def predict(self, box_area_ratio):
        """
        Predict distance from a single box_area_ratio value.

        Args:
            box_area_ratio (float): Bounding box area / frame area.

        Returns:
            float: Estimated distance in centimeters.
        """
        if not self._is_trained and self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor([[box_area_ratio]])
                y = self.model(x)
                distance = float(y.item())
        else:
            # Fallback: use stored regression coefficients
            distance = self._fallback_predict(box_area_ratio)

        # Clamp to reasonable range
        distance = max(10.0, min(500.0, distance))
        return distance

    def predict_batch(self, box_area_ratios):
        """
        Predict distances for a batch of box_area_ratio values.

        Args:
            box_area_ratios (array-like): Array of box_area_ratio values.

        Returns:
            np.ndarray: Estimated distances in centimeters.
        """
        if not self._is_trained and self.model is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")

        if HAS_TORCH and self.model is not None:
            self.model.eval()
            with torch.no_grad():
                x = torch.FloatTensor(np.array(box_area_ratios).reshape(-1, 1))
                y = self.model(x)
                distances = y.numpy().flatten()
        else:
            distances = np.array([self._fallback_predict(r) for r in box_area_ratios])

        distances = np.clip(distances, 10.0, 500.0)
        return distances

    def save(self, path=None):
        """Save the trained model to disk."""
        save_path = path or self.model_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        save_data = {
            'n_mfs': self.n_mfs,
            'input_min': self._input_min,
            'input_max': self._input_max,
            'output_min': self._output_min,
            'output_max': self._output_max,
            'is_trained': self._is_trained,
        }

        if HAS_TORCH and self.model is not None:
            save_data['state_dict'] = self.model.state_dict()
            save_data['mf_params'] = self.model.get_mf_params()

        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"[ANFIS] Model saved to: {save_path}")

    def load(self, path=None):
        """Load a trained model from disk."""
        load_path = path or self.model_path

        if not os.path.exists(load_path):
            raise FileNotFoundError(f"ANFIS model not found at: {load_path}")

        with open(load_path, 'rb') as f:
            save_data = pickle.load(f)

        self.n_mfs = save_data['n_mfs']
        self._input_min = save_data['input_min']
        self._input_max = save_data['input_max']
        self._output_min = save_data['output_min']
        self._output_max = save_data['output_max']
        self._is_trained = save_data['is_trained']

        if HAS_TORCH and 'state_dict' in save_data:
            self.model = ANFISModel(
                n_mfs=self.n_mfs,
                input_min=self._input_min,
                input_max=self._input_max,
            )
            self.model.load_state_dict(save_data['state_dict'])
            self.model.eval()

        print(f"[ANFIS] Model loaded from: {load_path}")

    def _fallback_predict(self, box_area_ratio):
        """
        Fallback distance estimation using inverse-square-root heuristic.
        Used when PyTorch is not available or model isn't trained.
        
        Based on pinhole camera model: distance ∝ 1 / sqrt(area_ratio)
        """
        if box_area_ratio <= 0:
            return 300.0
        # Approximate calibration: area_ratio ~0.15 ≈ 30cm, ~0.002 ≈ 300cm
        distance = 12.0 / (np.sqrt(box_area_ratio) + 0.01)
        return float(np.clip(distance, 10.0, 500.0))


# ─── Calibration Data Helper ──────────────────────────────────────────────────

def create_sample_calibration_data():
    """
    Generate synthetic calibration data for testing.
    
    In real use, this data comes from photographing objects at known distances
    and measuring the resulting box_area_ratio.
    
    Approximate relationship (pinhole model): area_ratio ∝ 1/distance²
    So: box_area_ratio ≈ k / distance²
    
    Returns:
        tuple: (box_area_ratios, true_distances_cm)
    """
    # Simulate: object at various distances, with noise
    distances = np.array([20, 30, 40, 50, 60, 80, 100, 120, 150, 180, 200, 250, 300],
                         dtype=np.float32)

    # k calibrated so that at 30cm, area_ratio ≈ 0.10 (typical for a phone)
    k = 0.10 * (30 ** 2)  # k = 90

    # Generate multiple samples per distance with noise
    all_ratios = []
    all_distances = []

    np.random.seed(42)
    for d in distances:
        n_samples = 5  # 5 shots per distance
        for _ in range(n_samples):
            noise = np.random.normal(1.0, 0.08)  # 8% noise
            ratio = (k / (d ** 2)) * noise
            ratio = np.clip(ratio, 0.0001, 1.0)
            all_ratios.append(ratio)
            all_distances.append(d)

    return np.array(all_ratios), np.array(all_distances)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ANFIS Distance Estimator — Training Demo ===\n")

    # Generate synthetic calibration data
    ratios, distances = create_sample_calibration_data()
    print(f"Calibration data: {len(ratios)} samples")
    print(f"  Area ratios range: [{ratios.min():.4f}, {ratios.max():.4f}]")
    print(f"  Distances range:   [{distances.min():.0f}, {distances.max():.0f}] cm\n")

    # Train
    estimator = ANFISDistanceEstimator(n_mfs=5)
    history = estimator.train(ratios, distances, epochs=2000, lr=0.01)

    # Test predictions
    print("\n--- Test Predictions ---")
    test_ratios = [0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.001]
    for r in test_ratios:
        pred = estimator.predict(r)
        print(f"  box_area_ratio={r:.4f} → predicted distance={pred:.1f} cm")

    # Save
    estimator.save()

    # Reload and verify
    print("\n--- Reload and Verify ---")
    estimator2 = ANFISDistanceEstimator()
    estimator2.load()
    for r in test_ratios:
        pred = estimator2.predict(r)
        print(f"  box_area_ratio={r:.4f} → predicted distance={pred:.1f} cm")
