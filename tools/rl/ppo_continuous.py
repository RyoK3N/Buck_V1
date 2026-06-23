"""
tools.rl.ppo_continuous
───────────────────────
PPO with an LSTM encoder and a tanh-Gaussian policy over continuous actions in
[0, 1] (target position fraction).

Design choices, with reasons:

  * **Windowed BPTT instead of stateful LSTM.** Every act/update consumes the
    last K=10 states as a sequence. Hidden state is recomputed from the window
    each time. This loses long-range temporal credit beyond K bars but keeps
    rollout collection trivially parallelizable and PPO minibatches
    independent — which is what matters for stable on-policy training on a
    single CPU.

  * **Tanh-Gaussian policy.** The agent emits (μ, log σ), samples z ~ N(μ, σ),
    then squashes via (tanh(z) + 1) / 2 ∈ [0, 1]. log_prob is corrected for
    the tanh Jacobian. This avoids the clip-bias artifact of Normal+clip and
    gives clean gradients near the boundaries.

  * **State-independent log σ.** A single learnable parameter (per action
    dim). The state-dependent version trains harder for negligible gain on
    1-D action spaces. Easy to swap later.

  * **GAE-λ advantages, advantage normalization, clipped surrogate, value
    clipping, entropy bonus.** Standard PPO-Clip recipe (Schulman 2017).

  * **No replay buffer** — PPO is on-policy. Rollouts collected via
    `collect_rollout(env)` are consumed by `update()` and discarded.

Checkpoint format includes `algorithm = "ppo_continuous"` so `load_agent()`
dispatches correctly.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .features import STATE_DIM


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")

ACTION_DIM = 1
DEFAULT_WINDOW = 10
LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


# ═══════════════════════════════════════════════════════════════════════════
#  Network
# ═══════════════════════════════════════════════════════════════════════════

class PPOContinuousNetwork(nn.Module):
    """LSTM encoder + tanh-Gaussian actor + value critic."""

    def __init__(self, state_dim: int = STATE_DIM, hidden_dim: int = 128,
                 lstm_layers: int = 1, action_dim: int = ACTION_DIM):
        super().__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.action_dim = action_dim

        self.lstm = nn.LSTM(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
        )
        # Two-layer MLP heads on top of the final hidden state.
        self.actor_trunk = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mu = nn.Linear(hidden_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -0.5))  # σ ≈ 0.6

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        """seq: (B, T, state_dim) → (B, hidden_dim) last hidden state."""
        _, (h_n, _) = self.lstm(seq)
        return h_n[-1]  # last layer

    def forward(self, seq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(seq)
        mu = self.actor_mu(self.actor_trunk(h))
        log_std = self.actor_log_std.expand_as(mu).clamp(LOG_STD_MIN, LOG_STD_MAX)
        value = self.critic(h).squeeze(-1)
        return mu, log_std, value


def _squash(z: torch.Tensor) -> torch.Tensor:
    """tanh-squashed action mapped from [-1, 1] → [0, 1]."""
    return 0.5 * (torch.tanh(z) + 1.0)


def _action_log_prob(z: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """log π(a|s) with tanh + half-range Jacobian correction.

    a = 0.5 * (tanh(z) + 1)   ⇒   log|da/dz| = log(0.5) + log(1 - tanh²(z))
    """
    std = log_std.exp()
    base = -0.5 * (((z - mu) / std) ** 2) - log_std - 0.5 * math.log(2 * math.pi)
    log_jac = math.log(0.5) + torch.log(1 - torch.tanh(z) ** 2 + 1e-6)
    return (base - log_jac).sum(-1)


# ═══════════════════════════════════════════════════════════════════════════
#  Agent
# ═══════════════════════════════════════════════════════════════════════════

class PPOContinuousAgent:
    algorithm = "ppo_continuous"

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hidden_dim: int = 128,
        lstm_layers: int = 1,
        window: int = DEFAULT_WINDOW,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 64,
        rollout_steps: int = 256,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lstm_layers = lstm_layers
        self.window = window
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.rollout_steps = rollout_steps

        self.device = DEVICE
        self.net = PPOContinuousNetwork(state_dim, hidden_dim, lstm_layers).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.train_steps = 0

        # Rolling state window — populated as the env feeds states in.
        self._window_buf: List[np.ndarray] = []

    # ── Acting ──────────────────────────────────────────────────────────────

    def reset_window(self) -> None:
        self._window_buf = []

    def _push_state(self, state: np.ndarray) -> torch.Tensor:
        self._window_buf.append(np.asarray(state, dtype=np.float32))
        if len(self._window_buf) > self.window:
            self._window_buf = self._window_buf[-self.window:]
        # Pad at the start by repeating the first state so seq length is constant.
        if len(self._window_buf) < self.window:
            pad = [self._window_buf[0]] * (self.window - len(self._window_buf))
            seq = np.stack(pad + self._window_buf, axis=0)
        else:
            seq = np.stack(self._window_buf, axis=0)
        return torch.tensor(seq, dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, T, D)

    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[float, Dict[str, float]]:
        """Returns (continuous_action in [0,1], info_dict with log_prob, value, z, mu, std).

        Maintains the rolling window internally — call reset_window() between
        episodes (the training loop does this for you).
        """
        seq = self._push_state(state)
        mu, log_std, value = self.net(seq)
        if eval_mode:
            z = mu
            log_prob = _action_log_prob(z, mu, log_std).item()
        else:
            std = log_std.exp()
            z = mu + std * torch.randn_like(std)
            log_prob = _action_log_prob(z, mu, log_std).item()
        action = float(_squash(z).item())
        return action, {
            "log_prob": log_prob,
            "value": float(value.item()),
            "z": float(z.item()),
            "mu": float(mu.item()),
            "log_std": float(log_std.item()),
        }

    # ── Rollout collection ──────────────────────────────────────────────────

    def collect_rollout(self, env, on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None) -> Dict[str, Any]:
        """Run the env for up to `rollout_steps` steps (or until done). Returns a
        dict of stacked arrays suitable for `update()`.

        The agent's window is *reset* at the start, so this assumes the caller
        also calls `env.reset()` if they want a fresh episode. For multi-rollout
        training over a single env, just keep calling this without resetting —
        the agent will continue from the last window state.
        """
        seqs: List[np.ndarray] = []
        actions: List[float] = []
        zs: List[float] = []
        log_probs: List[float] = []
        values: List[float] = []
        rewards: List[float] = []
        dones: List[float] = []

        if not self._window_buf:
            state = env.reset()
            self.reset_window()
        else:
            state = env._state()

        last_info: Dict[str, Any] = {}
        for t in range(self.rollout_steps):
            # Build seq input for this step BEFORE acting so we can store it
            # for the update (training needs the same input the policy saw).
            self._window_buf.append(np.asarray(state, dtype=np.float32))
            if len(self._window_buf) > self.window:
                self._window_buf = self._window_buf[-self.window:]
            if len(self._window_buf) < self.window:
                pad = [self._window_buf[0]] * (self.window - len(self._window_buf))
                seq_np = np.stack(pad + self._window_buf, axis=0)
            else:
                seq_np = np.stack(self._window_buf, axis=0)

            with torch.no_grad():
                seq_t = torch.tensor(seq_np, dtype=torch.float32, device=self.device).unsqueeze(0)
                mu, log_std, value = self.net(seq_t)
                std = log_std.exp()
                z = mu + std * torch.randn_like(std)
                log_prob = _action_log_prob(z, mu, log_std).item()
                action_val = float(_squash(z).item())

            next_state, reward, done, info = env.step(action_val)
            if on_step is not None:
                on_step(t, {"action": action_val, "reward": reward, "info": info})

            seqs.append(seq_np)
            actions.append(action_val)
            zs.append(float(z.item()))
            log_probs.append(log_prob)
            values.append(float(value.item()))
            rewards.append(float(reward))
            dones.append(1.0 if done else 0.0)
            last_info = {"next_state": next_state, "done": done}
            state = next_state
            if done:
                break

        # Bootstrap value for the final state
        with torch.no_grad():
            if last_info.get("done"):
                next_value = 0.0
            else:
                # Build window for the post-rollout state too
                self._window_buf.append(np.asarray(last_info["next_state"], dtype=np.float32))
                if len(self._window_buf) > self.window:
                    self._window_buf = self._window_buf[-self.window:]
                if len(self._window_buf) < self.window:
                    pad = [self._window_buf[0]] * (self.window - len(self._window_buf))
                    last_seq = np.stack(pad + self._window_buf, axis=0)
                else:
                    last_seq = np.stack(self._window_buf, axis=0)
                _, _, v = self.net(torch.tensor(last_seq, dtype=torch.float32, device=self.device).unsqueeze(0))
                next_value = float(v.item())

        return {
            "seqs": np.stack(seqs, axis=0),
            "actions": np.asarray(actions, dtype=np.float32),
            "zs": np.asarray(zs, dtype=np.float32),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "next_value": next_value,
            "episode_done": bool(last_info.get("done", False)),
            "n_steps": len(rewards),
        }

    # ── Update ──────────────────────────────────────────────────────────────

    def _compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(T)):
            v_next = next_value if t == T - 1 else values[t + 1]
            non_terminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * v_next * non_terminal - values[t]
            last_gae = delta + self.gamma * self.lam * non_terminal * last_gae
            adv[t] = last_gae
        returns = adv + values
        return adv, returns

    def update(self, rollout: Dict[str, Any]) -> Dict[str, float]:
        T = rollout["n_steps"]
        if T == 0:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                    "entropy": 0.0, "mean_advantage": 0.0, "mean_return": 0.0}

        seqs = torch.tensor(rollout["seqs"], dtype=torch.float32, device=self.device)        # (T, W, D)
        actions = torch.tensor(rollout["actions"], dtype=torch.float32, device=self.device)  # (T,)
        zs = torch.tensor(rollout["zs"], dtype=torch.float32, device=self.device).unsqueeze(-1)  # (T, 1)
        old_log_probs = torch.tensor(rollout["log_probs"], dtype=torch.float32, device=self.device)
        old_values = torch.tensor(rollout["values"], dtype=torch.float32, device=self.device)

        adv_np, ret_np = self._compute_gae(
            rollout["rewards"], rollout["values"], rollout["dones"], rollout["next_value"]
        )
        advantages = torch.tensor(adv_np, dtype=torch.float32, device=self.device)
        returns = torch.tensor(ret_np, dtype=torch.float32, device=self.device)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        total_pol = 0.0
        total_val = 0.0
        total_ent = 0.0
        n_minibatches = 0

        indices = np.arange(T)
        for _ in range(self.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, T, self.batch_size):
                mb_idx = indices[start:start + self.batch_size]
                if len(mb_idx) < 2:
                    continue
                mb_seqs = seqs[mb_idx]
                mb_zs = zs[mb_idx]
                mb_old_lp = old_log_probs[mb_idx]
                mb_old_v = old_values[mb_idx]
                mb_adv = advantages[mb_idx]
                mb_ret = returns[mb_idx]

                mu, log_std, value = self.net(mb_seqs)
                new_log_probs = _action_log_prob(mb_zs, mu, log_std)
                # Differential entropy of the underlying Normal (independent of state).
                entropy = (log_std + 0.5 * math.log(2 * math.pi * math.e)).sum(-1).mean()

                ratio = torch.exp(new_log_probs - mb_old_lp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss with PPO-style value clipping
                value_pred_clipped = mb_old_v + torch.clamp(value - mb_old_v, -self.value_clip, self.value_clip)
                v_loss_unclipped = (value - mb_ret) ** 2
                v_loss_clipped = (value_pred_clipped - mb_ret) ** 2
                value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.train_steps += 1

                total_loss += float(loss.item())
                total_pol += float(policy_loss.item())
                total_val += float(value_loss.item())
                total_ent += float(entropy.item())
                n_minibatches += 1

        nm = max(n_minibatches, 1)
        return {
            "loss": total_loss / nm,
            "policy_loss": total_pol / nm,
            "value_loss": total_val / nm,
            "entropy": total_ent / nm,
            "mean_advantage": float(advantages.mean().item()),
            "mean_return": float(returns.mean().item()),
        }

    # ── Persistence ─────────────────────────────────────────────────────────

    def save(self, model_id: str) -> str:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        torch.save({
            "algorithm": self.algorithm,
            "state_dict": self.net.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "train_steps": self.train_steps,
            "input_dim": self.state_dim,
            "hidden_dim": self.hidden_dim,
            "lstm_layers": self.lstm_layers,
            "window": self.window,
            "lr": self.lr,
            "gamma": self.gamma,
            "lam": self.lam,
            "clip_epsilon": self.clip_epsilon,
            "entropy_coef": self.entropy_coef,
            "value_coef": self.value_coef,
            "rollout_steps": self.rollout_steps,
            "action_dim": ACTION_DIM,
        }, path)
        return str(path)

    def load(self, model_id: str) -> bool:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if ckpt.get("algorithm") != self.algorithm or "state_dict" not in ckpt:
            return False  # checkpoint belongs to a different agent type
        self.state_dim = ckpt.get("input_dim", self.state_dim)
        self.hidden_dim = ckpt.get("hidden_dim", self.hidden_dim)
        self.lstm_layers = ckpt.get("lstm_layers", self.lstm_layers)
        self.window = ckpt.get("window", self.window)
        self.lr = ckpt.get("lr", self.lr)
        self.gamma = ckpt.get("gamma", self.gamma)
        self.lam = ckpt.get("lam", self.lam)
        self.clip_epsilon = ckpt.get("clip_epsilon", self.clip_epsilon)
        self.entropy_coef = ckpt.get("entropy_coef", self.entropy_coef)
        self.value_coef = ckpt.get("value_coef", self.value_coef)
        self.rollout_steps = ckpt.get("rollout_steps", self.rollout_steps)

        self.net = PPOContinuousNetwork(
            state_dim=self.state_dim,
            hidden_dim=self.hidden_dim,
            lstm_layers=self.lstm_layers,
        ).to(self.device)
        self.net.load_state_dict(ckpt["state_dict"])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        try:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (KeyError, ValueError):
            pass  # state-dict shape mismatch on re-init is fine
        self.train_steps = ckpt.get("train_steps", 0)
        self.reset_window()
        return True


if __name__ == "__main__":  # tiny smoke test
    import yfinance as yf
    import pandas as pd
    from .env import TradingEnvironment

    df = yf.download("AAPL", period="6mo", interval="1d", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    env = TradingEnvironment(df)
    agent = PPOContinuousAgent(rollout_steps=32, update_epochs=2, batch_size=8)
    env.reset()
    agent.reset_window()
    rollout = agent.collect_rollout(env)
    stats = agent.update(rollout)
    print(f"rollout steps: {rollout['n_steps']}, episode_done: {rollout['episode_done']}")
    print(f"update stats: {stats}")
    print(f"env summary: {env.summary()}")
