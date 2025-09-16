import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreEntropyDiscreteDiffusino(nn.Module):
    def __init__(self,
                 num_bins,
                 denoise_fn,
                 num_timesteps=1000,
                 schedule="cosine",
                 device="cpu"):
        super().__init__()
        self.num_bins = num_bins
        self.num_features = len(num_bins)
        self.denoise_fn = denoise_fn
        self.num_timesteps = num_timesteps
        self.device = device

        # score matrix : each feature value -> label/target alignment
        # in initial uniform -> propensity transition
        self.score_matrices = [
            torch.ones((k, k), device=device) / k for k in num_bins
        ]

        # noise schedule
        self.schedule = schedule
        self.sigmas = self._make_schedule(schedule, num_timesteps).to(device)

    def _make_schedule(self, name, T):
        if name == "linear":
            return torch.linspace(1e-4, 1.0, T)
        elif name == "cosine":
            steps = torch.arange(T + 1, dtype=torch.float64)
            alphas_cumprod = torch.cos((steps / T + 0.008) / 1.008 * torch.pi / 2) ** 2
            sigmas = 1. - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return sigmas.float()
        else:
            raise NotImplementedError

    def q_forward(self, log_x, t):
        """
        Forward diffusion step:
        p_t+1 = (1 - sigma_t) * p_t + sigma_t * (Q @ p_t)
        log_x : one-hot(log) representation
        t : current timestep
        """
        sigma_t = self.sigmas[t]

        out = []
        offset = 0
        for f, k in enumerate(self.num_bins):
            log_p = log_x[:, offset:offset+k]   # [B, K]
            Q = self.score_matrices[f]          # [K, K]
            p = log_p.exp()
            p_next = (1 - sigma_t) * p + sigma_t * (p @ Q)
            out.append(torch.log(p_next + 1e-8))
            offset += k
        return torch.cat(out, dim=1)

    def q_sample(self, log_x0, t):
        """
        Sample x_t given x0 by applying generator forward
        """
        log_p = log_x0
        for step in range(t):
            log_p = self.q_forward(log_p, step)
        return self.log_sample_categorical(log_p)

    def log_sample_categorical(self, log_probs):
        """
        Gumbel-max trick
        """
        B, D = log_probs.shape
        out = []
        offset = 0
        for k in self.num_bins:
            lp = log_probs[:, offset:offset+k]
            gumbel = -torch.log(-torch.log(torch.rand_like(lp) + 1e-30) + 1e-30)
            sample = (lp + gumbel).argmax(dim=1)
            onehot = F.one_hot(sample, num_classes=k).float()
            out.append(torch.log(onehot + 1e-8))
            offset += k
        return torch.cat(out, dim=1)

    def p_reverse(self, model_out, log_x_t, t):
        """
        Reverse step:
        use model output (predicted x0 logits) + SEDD time reversal
        """
        # predict start distribution
        log_x0_pred = self.predict_start(model_out, log_x_t, t)

        # construct reverse generator Qbar
        # q_posterior = f(log_x0_pred, log_x_t)
        return log_x0_pred

    def predict_start(self, model_out, log_x_t, t):
        """
        Turn model logits into log-probabilities
        """
        out = []
        offset = 0
        for f, k in enumerate(self.num_bins):
            logits = model_out[:, offset:offset+k]
            log_probs = F.log_softmax(logits, dim=1)
            out.append(log_probs)
            offset += k
        return torch.cat(out, dim=1)

    def mixed_loss(self, x, y=None):
        """
        Training loss (SEDD style): forward sample + reverse denoising
        """
        B = x.shape[0]
        device = x.device

        # One-hot representation
        log_x0 = []
        offset = 0
        for f, k in enumerate(self.num_bins):
            onehot = F.one_hot(x[:, f], num_classes=k).float()
            log_x0.append(torch.log(onehot + 1e-8))
            offset += k
        log_x0 = torch.cat(log_x0, dim=1)

        # sample time
        t = torch.randint(0, self.num_timesteps, (B,), device=device).long()

        # forward diffusion
        log_x_t = self.q_sample(log_x0, t.max())

        # model prediction
        model_out = self.denoise_fn(log_x_t, t)

        # loss: cross-entropy between predicted start vs true start
        log_x0_pred = self.predict_start(model_out, log_x_t, t)
        loss = -(log_x0 * log_x0_pred.exp()).sum(dim=1).mean()

        return loss