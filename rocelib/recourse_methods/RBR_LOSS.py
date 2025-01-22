import torch


def l2_projection(x, radius):
    """
    Euclidean projection onto an L2-ball
    """
    norm = torch.linalg.norm(x, ord=2, axis=-1)
    return (radius * 1 / torch.max(radius, norm)).unsqueeze(1) * x


class OptimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim, epsilon_op, sigma, device="cpu"):
        super(OptimisticLikelihood, self).__init__()
        self.device = device
        self.x_dim = x_dim.to(self.device)
        self.epsilon_op = epsilon_op.to(self.device)
        self.sigma = sigma.to(self.device)

    @torch.no_grad()
    def projection(self, v):
        v = v.clone()
        v = torch.max(v, torch.tensor(0.0, device=self.device))
        return l2_projection(v, self.epsilon_op).to(self.device)

    def forward(self, v, x, x_feas):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = v[..., 1] + self.sigma
        p = self.x_dim

        L = torch.log(d) + (c - v[..., 0]) ** 2 / (2 * d ** 2) + (p - 1) * torch.log(self.sigma)

        v_grad = torch.zeros_like(v, device=self.device)
        v_grad[..., 0] = -(c - v[..., 0]) / d ** 2
        v_grad[..., 1] = 1 / d - (c - v[..., 0]) ** 2 / d ** 3

        return L, v_grad

    def optimize(self, x, x_feas, max_iter=1000):
        v = torch.zeros([x.shape[0], 2], device=self.device)
        min_loss = float('inf')
        num_stable_iter = 0

        for _ in range(max_iter):
            F, grad = self.forward(v, x, x_feas)
            v = self.projection(v - 1 / torch.sqrt(torch.tensor(max_iter, device=self.device)) * grad)

            loss_sum = F.sum().item()
            if min_loss - loss_sum <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= 10:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)

        return v


class PessimisticLikelihood(torch.nn.Module):
    def __init__(self, x_dim, epsilon_pe, sigma, device="cpu"):
        super(PessimisticLikelihood, self).__init__()
        self.device = device
        self.epsilon_pe = epsilon_pe.to(self.device)
        self.sigma = sigma.to(self.device)
        self.x_dim = x_dim.to(self.device)

    @torch.no_grad()
    def projection(self, u):
        u = u.clone()
        u = torch.max(u, torch.tensor(0.0, device=self.device))
        return l2_projection(u, self.epsilon_pe / torch.sqrt(self.x_dim)).to(self.device)

    def forward(self, u, x, x_feas, zeta=1e-6):
        c = torch.linalg.norm(x - x_feas, axis=-1)
        d = u[..., 1] + self.sigma
        p = self.x_dim
        sqrt_p = torch.sqrt(p)
        f = torch.sqrt((zeta + self.epsilon_pe ** 2 - p * u[..., 0] ** 2 - u[..., 1] ** 2) / (p - 1))

        L = -torch.log(d) - (c + sqrt_p * u[..., 0]) ** 2 / (2 * d ** 2) - (p - 1) * torch.log(f + self.sigma)

        u_grad = torch.zeros_like(u, device=self.device)
        u_grad[..., 0] = -sqrt_p * (c + sqrt_p * u[..., 0]) / d ** 2 - (p * u[..., 0]) / (f * (f + self.sigma))
        u_grad[..., 1] = -1 / d + (c + sqrt_p * u[..., 0]) ** 2 / d ** 3 + u[..., 1] / (f * (f + self.sigma))

        return L, u_grad

    def optimize(self, x, x_feas, max_iter=1000):
        u = torch.zeros([x.shape[0], 2], device=self.device)
        min_loss = float('inf')
        num_stable_iter = 0

        for _ in range(max_iter):
            F, grad = self.forward(u, x, x_feas)
            u = self.projection(u - 1 / torch.sqrt(torch.tensor(max_iter, device=self.device)) * grad)

            loss_sum = F.sum().item()
            if min_loss - loss_sum <= 1e-10:
                num_stable_iter += 1
                if num_stable_iter >= 10:
                    break
            else:
                num_stable_iter = 0
            min_loss = min(min_loss, loss_sum)

        return u


class RBRLoss(torch.nn.Module):
    def __init__(self, X_feas, X_feas_pos, X_feas_neg, epsilon_op, epsilon_pe, sigma, device="cpu"):
        super(RBRLoss, self).__init__()
        self.device = device
        self.X_feas = X_feas.to(self.device)
        self.X_feas_pos = X_feas_pos.to(self.device)
        self.X_feas_neg = X_feas_neg.to(self.device)

        self.epsilon_op = torch.tensor(epsilon_op).to(self.device)
        self.epsilon_pe = torch.tensor(epsilon_pe).to(self.device)
        self.sigma = torch.tensor(sigma).to(self.device)
        self.x_dim = torch.tensor(X_feas.shape[-1]).to(self.device)

        self.op_likelihood = OptimisticLikelihood(self.x_dim, self.epsilon_op, self.sigma, self.device)
        self.pe_likelihood = PessimisticLikelihood(self.x_dim, self.epsilon_pe, self.sigma, self.device)

    def forward(self, x):
        u = self.pe_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_pos.shape[0], -1]),
            self.X_feas_pos
        )

        F_pe = torch.stack([
            self.pe_likelihood.forward(u, x.expand([self.X_feas_pos.shape[0], -1]), self.X_feas_pos)[0]
            for _ in range(u.shape[0])
        ])
        denom = torch.logsumexp(F_pe, -1)

        v = self.op_likelihood.optimize(
            x.detach().clone().expand([self.X_feas_neg.shape[0], -1]),
            self.X_feas_neg
        )

        F_op = torch.stack([
            self.op_likelihood.forward(v, x.expand([self.X_feas_neg.shape[0], -1]), self.X_feas_neg)[0]
            for _ in range(v.shape[0])
        ])
        numer = torch.logsumexp(-F_op, -1)

        return numer - denom, denom, numer