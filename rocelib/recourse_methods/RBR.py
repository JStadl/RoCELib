import pandas as pd
import torch
import numpy as np
from sklearn.utils import check_random_state
from rocelib.recourse_methods.RecourseGenerator import RecourseGenerator
from rocelib.tasks.Task import Task
from rocelib.recourse_methods.RBR_LOSS import RBRLoss


class RBR(RecourseGenerator):
    """
    RBR (Region-Based Recourse) implementation for generating counterfactual explanations.
    """

    def __init__(self, ct: Task, num_samples=100, perturb_radius=0.1, delta_plus=0.1,
                 sigma=0.1, epsilon_op=0.1, epsilon_pe=0.1, max_iter=500, random_state=None):
        super().__init__(ct)
        self.num_samples = num_samples
        self.perturb_radius = perturb_radius
        self.delta_plus = delta_plus
        self.sigma = sigma
        self.epsilon_op = epsilon_op
        self.epsilon_pe = epsilon_pe
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feasible = True

    def dist(self, a, b):
        """Calculate L1 distance between points."""
        if isinstance(a, pd.DataFrame):
            a = torch.tensor(a.values, dtype=torch.float32)
        if isinstance(b, pd.DataFrame):
            b = torch.tensor(b.values, dtype=torch.float32)
        return torch.linalg.norm(a - b, ord=1, dim=-1)

    def find_x_boundary(self, x, target_label=1):
        """Find a point on the decision boundary."""
        train_data = self.task.training_data.data.drop(columns=["target"], errors='ignore')
        x_tensor = torch.tensor(x.values if isinstance(x, pd.DataFrame) else x, dtype=torch.float32)
        train_tensor = torch.tensor(train_data.values, dtype=torch.float32).to(self.device)
        x_tensor = x_tensor.to(self.device)

        # Get model predictions for training data
        train_preds = torch.tensor([
            self.task.model.predict_single(row)
            for _, row in train_data.iterrows()
        ]).to(self.device)

        # Find points of target class
        mask = train_preds == target_label
        target_points = train_tensor[mask]

        if len(target_points) == 0:
            return None, torch.tensor(float('inf'))

        # Calculate distances
        distances = self.dist(x_tensor, target_points)
        min_idx = torch.argmin(distances)

        return target_points[min_idx], distances[min_idx]

    def uniform_ball(self, x, radius, n):
        """Generate uniform samples from a ball centered at x."""
        d = len(x)
        x_numpy = x.cpu().numpy() if isinstance(x, torch.Tensor) else x

        # Generate random directions
        directions = self.random_state.randn(n, d)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

        # Generate random radii
        radii = self.random_state.random(n) ** (1.0 / d)
        points = x_numpy + (directions * radii[:, np.newaxis]) * radius

        return torch.tensor(points, dtype=torch.float32).to(self.device)

    def feasible_set(self, x, radius, num_samples):
        """Generate feasible set of points around x."""
        return self.uniform_ball(x, radius, num_samples)

    def projection(self, x, delta):
        """Project x onto L1 ball of radius delta."""
        if torch.linalg.norm(x, ord=1) <= delta:
            return x

        # Sort absolute values
        x_abs = torch.abs(x)
        x_sort, _ = torch.sort(x_abs, descending=True)

        # Calculate cumulative sums
        cssv = torch.cumsum(x_sort, 0)
        rho = torch.nonzero(x_sort * torch.arange(1, len(x) + 1).to(self.device) > (cssv - delta))[-1]

        theta = (cssv[rho] - delta) / (rho + 1.0)
        proj = torch.clamp(x_abs - theta, min=0) * torch.sign(x)

        return proj

    def optimize(self, x, delta, loss_fn):
        """Optimize the counterfactual."""
        x_t = x.detach().clone()
        x_t.requires_grad_(True)
        min_loss = float('inf')
        num_stable_iter = 0

        for _ in range(self.max_iter):
            if x_t.grad is not None:
                x_t.grad.data.zero_()

            F, _, _ = loss_fn(x_t)
            F.backward()

            if self.dist(x_t.detach(), x) >= delta:
                break

            with torch.no_grad():
                x_new = x_t - 1 / torch.sqrt(torch.tensor(1e3, device=self.device)) * x_t.grad
                x_new = self.projection(x_new - x, delta) + x
                x_t.data = x_new

            loss_sum = F.sum().item()
            if min_loss - loss_sum <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= 10:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)

        return F, x_t.detach()

    def _generation_method(self, instance, column_name="target", neg_value=0, **kwargs):
        """Generate counterfactual explanation for an instance."""
        # Convert instance to numpy if needed
        if isinstance(instance, pd.DataFrame):
            x0 = torch.tensor(instance.iloc[0].values, dtype=torch.float32)
        else:
            x0 = torch.tensor(instance.values, dtype=torch.float32)

        x0 = x0.to(self.device)

        # Find boundary point
        x_b, delta_base = self.find_x_boundary(x0, target_label=1 - neg_value)
        if x_b is None:
            return pd.DataFrame()

        delta = delta_base + self.delta_plus

        # Generate feasible set
        X_feas = self.feasible_set(x_b, self.perturb_radius, self.num_samples)

        # Get predictions for feasible set
        y_feas = torch.tensor([
            self.task.model.predict_single(x.cpu())
            for x in X_feas
        ]).to(self.device)

        # Split feasible set into positive and negative
        X_feas_pos = X_feas[y_feas == 1 - neg_value]
        X_feas_neg = X_feas[y_feas == neg_value]

        if len(X_feas_pos) == 0 or len(X_feas_neg) == 0:
            # If there are no points in either positive or negative set, return an empty DataFrame
            return pd.DataFrame()

        # Step 5: Compute RBR loss
        rbr_loss = RBRLoss(X_feas, X_feas_pos, X_feas_neg,
                           epsilon_op=self.epsilon_op, epsilon_pe=self.epsilon_pe,
                           sigma=self.sigma, device=self.device)

        def loss_fn(x):
            return rbr_loss(x)

        # Step 6: Optimize to find the counterfactual
        _, x_cf = self.optimize(x0, delta, loss_fn)

        # Step 7: Convert result to DataFrame
        result = pd.DataFrame(x_cf.cpu().numpy().reshape(1, -1), columns=instance.columns if isinstance(instance, pd.DataFrame) else None)

        return result