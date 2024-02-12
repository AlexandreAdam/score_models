from score_models.sde import VESDE, VPSDE, TSVESDE
import numpy as np
import torch

def get_trajectories(sde, B=10, N=100, x0=5):
    dt = 1/N
    t = torch.zeros(B) + sde.epsilon
    x0 = torch.ones(B) * x0
    x = torch.clone(x0)
    trajectories = [x]
    marginal_samples = [x]
    for step in range(N):
        t += dt
        f = sde.drift(t, x)
        g = sde.diffusion(t, x)
        dw = torch.randn_like(x) * dt**(1/2)
        x = x + f * dt + g * dw
        trajectories.append(x)
        marginal_samples.append(sde.sample_marginal(t, x0))
    trajectories = np.stack(trajectories)
    marginal_samples = np.stack(marginal_samples)
    return trajectories, marginal_samples
        


# Visual test that marginals of the trajectories are as expected.
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    B = 100
    N = 1000
    x0 = 1e2
    sde1 = VESDE(sigma_min=1e-1, sigma_max=100)
    sde2 = VPSDE(beta_min=1e-2, beta_max=20)
    sde3 = TSVESDE(sigma_min=1e-6, sigma_max=1e9, t_star=0.4, beta=30, beta_fn="relu")
    sde4 = TSVESDE(sigma_min=1e-4, sigma_max=1e6, t_star=0.4, beta=20, beta_fn="silu")
    sde5 = TSVESDE(sigma_min=1e-4, sigma_max=1e6, t_star=0.4, beta=20, beta_fn="hardswish")
    
    text = ["", "", "relu", "silu", "hardswish"]
    for i, sde in enumerate([sde1, sde2, sde3, sde4, sde5]):
        trajectories, marginal_samples = get_trajectories(sde, B, N, x0=x0)
        
        fig, axs = plt.subplots(2, 2, figsize=(8, 4), sharex=True)

        fig.suptitle(sde.__class__.__name__ + " " + text[i], y=0.96)
        axs[0, 0].set_title("Trajectories")
        axs[0, 1].set_title("Samples from the time marginals")
        axs[1, 0].set_xlabel("t")
        axs[1, 1].set_xlabel("t")
        axs[0, 0].set_ylabel("x")
        axs[1, 0].set_ylabel("x")
        t = np.linspace(0, 1, N+1)
        for b in range(B):
            axs[0, 0].plot(t, trajectories[:, b])
        
        axs[1, 0].plot(t, trajectories.std(axis=1), "k-", alpha=0.5, label=r"Empirical $\sigma(t)$")
        axs[1, 0].plot(t, trajectories.mean(axis=1), "r-", alpha=0.5, label=r"Empirical $\mu(t)$")

        mu, sigma = sde.marginal_prob_scalars(torch.tensor(t))
        axs[1, 0].plot(t, sigma, "k--", label=r"Expected $\sigma(t)$")
        axs[1, 0].plot(t, mu * x0, "r-", label=r"Expected $\mu(t)$")
        # axs[1, 0].legend()
        
        
        for b in range(B):
            axs[0, 1].plot(t, marginal_samples[:, b])
        
        axs[1, 1].plot(t, marginal_samples.std(axis=1), "k-", alpha=0.5, label=r"Empirical $\sigma(t)$")
        axs[1, 1].plot(t, marginal_samples.mean(axis=1), "r-", alpha=0.5,label=r"Empirical $\mu(t)$")
        axs[1, 1].plot(t, sigma, "k--", label=r"Expected $\sigma(t)$")
        axs[1, 1].plot(t, mu * x0, "r-", label=r"Expected $\mu(t)$")
        axs[1, 1].legend(bbox_to_anchor=(1.1, 1.05))
        fig.tight_layout()
    plt.show()
