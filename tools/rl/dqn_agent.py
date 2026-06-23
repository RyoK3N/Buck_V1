from __future__ import annotations
import math
import random
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


WEIGHTS_DIR = Path(__file__).resolve().parent / "weights"
WEIGHTS_DIR.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")


# ═══════════════════════════════════════════════════════════════
#  Shared state extraction
# ═══════════════════════════════════════════════════════════════

def extract_state(df: pd.DataFrame, idx: int, position: int, cash_ratio: float, window: int = 10) -> np.ndarray:
    close = df['Close'].values
    volume = df['Volume'].values
    high = df['High'].values
    low = df['Low'].values

    prices = close[max(0, idx - window + 1):idx + 1]
    vols = volume[max(0, idx - window + 1):idx + 1]

    if len(prices) < window:
        pad = window - len(prices)
        prices = np.pad(prices, (pad, 0), 'edge')
        vols = np.pad(vols, (pad, 0), 'edge')

    log_ret = np.log(prices[-1] / (prices[-2] + 1e-10)) if len(prices) >= 2 else 0.0
    momentum = (prices[-1] / (prices[0] + 1e-10)) - 1.0
    volatility = np.std(np.diff(np.log(prices + 1e-10))) if len(prices) >= 2 else 0.0

    gains = 0.0
    losses = 0.0
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        if diff > 0:
            gains += diff
        else:
            losses -= diff
    avg_gain = gains / (len(prices) - 1) if len(prices) > 1 else 0
    avg_loss = losses / (len(prices) - 1) if len(prices) > 1 else 1e-10
    rsi = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-10)))

    price_z = (prices[-1] - np.mean(prices)) / (np.std(prices) + 1e-10)
    vol_z = (vols[-1] - np.mean(vols)) / (np.std(vols) + 1e-10)
    high_low = (high[idx] - low[idx]) / (close[idx] + 1e-10) if idx < len(high) else 0.0

    close_ma = np.mean(prices)
    bb_upper = close_ma + 2 * np.std(prices)
    bb_lower = close_ma - 2 * np.std(prices)
    bb_pos = (prices[-1] - bb_lower) / (bb_upper - bb_lower + 1e-10)

    return np.array([
        price_z, vol_z, log_ret, momentum, volatility,
        rsi / 100.0, high_low, bb_pos,
        float(position), cash_ratio,
        prices[-1] / (prices[-2] + 1e-10) - 1.0 if len(prices) >= 2 else 0.0,
        np.mean(prices[-5:]) / (np.mean(prices[-10:]) + 1e-10) - 1.0 if len(prices) >= 10 else 0.0,
    ], dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
#  DQN
# ═══════════════════════════════════════════════════════════════

class DQN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    algorithm = "dqn"

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.998,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        tau: float = 0.005,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.tau = tau
        self.device = DEVICE

        self.policy_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net = DQN(input_dim, hidden_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_capacity)
        self.train_steps = 0

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        if not eval_mode and random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(s)
            return int(q_values.argmax().item())

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self) -> Optional[float]:
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([b[1] for b in batch], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor([b[2] for b in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float32, device=self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1, keepdim=True)
            next_q = self.target_net(next_states).gather(1, next_actions).squeeze()
            target_q = rewards + (1 - dones) * self.gamma * next_q

        loss = self.criterion(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        for tp, pp in zip(self.target_net.parameters(), self.policy_net.parameters()):
            tp.data.copy_(self.tau * pp.data + (1.0 - self.tau) * tp.data)

        self.train_steps += 1
        return loss.item()

    def end_episode(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, model_id: str) -> str:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        torch.save({
            'algorithm': self.algorithm,
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_steps': self.train_steps,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'lr': self.lr,
            'gamma': self.gamma,
        }, path)
        return str(path)

    def load(self, model_id: str) -> bool:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.input_dim = ckpt['input_dim']
        self.hidden_dim = ckpt['hidden_dim']
        self.output_dim = ckpt['output_dim']
        self.lr = ckpt.get('lr', self.lr)
        self.gamma = ckpt.get('gamma', self.gamma)
        self.policy_net = DQN(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.target_net = DQN(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.policy_net.load_state_dict(ckpt['policy_state_dict'])
        self.target_net.load_state_dict(ckpt['target_state_dict'])
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.epsilon = ckpt.get('epsilon', self.epsilon_min)
        self.train_steps = ckpt.get('train_steps', 0)
        self.target_net.eval()
        return True


# ═══════════════════════════════════════════════════════════════
#  PPO
# ═══════════════════════════════════════════════════════════════

class PPONetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.fc(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class PPOAgent:
    algorithm = "ppo"

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        clip_epsilon: float = 0.2,
        update_epochs: int = 10,
        batch_size: int = 64,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.lam = lam
        self.clip_epsilon = clip_epsilon
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.device = DEVICE
        self.train_steps = 0

        self.net = PPONetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self._trajectory: List[Dict] = []

    def get_action_and_value(self, state: np.ndarray, eval_mode: bool = False) -> Tuple[int, float, float, float]:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        probs = F.softmax(logits, dim=-1)
        if eval_mode:
            action = int(probs.argmax().item())
            log_prob = math.log(probs[0, action].item() + 1e-10)
            entropy = float(-(probs * torch.log(probs + 1e-10)).sum().item())
        else:
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
            action = int(a.item())
            log_prob = float(dist.log_prob(a).item())
            entropy = float(dist.entropy().item())
        return action, log_prob, entropy, float(value.item())

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        action, log_prob, entropy, value = self.get_action_and_value(state, eval_mode)
        if not eval_mode:
            self._trajectory.append({
                'state': state.copy() if hasattr(state, 'copy') else np.array(state),
                'action': action,
                'log_prob': log_prob,
                'value': value,
            })
        return action

    def remember(self, state, action, reward, next_state, done):
        if self._trajectory:
            self._trajectory[-1]['reward'] = reward
            self._trajectory[-1]['next_state'] = next_state
            self._trajectory[-1]['done'] = done

    def update(self, trajectories: List[Dict]) -> Dict[str, float]:
        states = torch.tensor(np.array([t['state'] for t in trajectories]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t['action'] for t in trajectories], dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor([t['log_prob'] for t in trajectories], dtype=torch.float32, device=self.device)
        rewards = torch.tensor([t['reward'] for t in trajectories], dtype=torch.float32, device=self.device)
        dones = torch.tensor([t['done'] for t in trajectories], dtype=torch.float32, device=self.device)
        next_state = trajectories[-1]['next_state']

        with torch.no_grad():
            _, next_value = self.net(torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0))
            next_value = next_value.squeeze().item()

        returns = []
        adv = 0
        for i in reversed(range(len(trajectories))):
            if i == len(trajectories) - 1:
                next_val = next_value
            else:
                _, val = self.net(states[i + 1:i + 2])
                next_val = val.item()
            delta = rewards[i] + self.gamma * next_val * (1 - dones[i].item()) - trajectories[i].get('value', 0)
            adv = delta + self.gamma * self.lam * (1 - dones[i].item()) * adv
            returns.insert(0, adv + trajectories[i].get('value', 0))
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages = returns - torch.tensor([t.get('value', 0) for t in trajectories], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss = 0.0
        n = len(states)
        indices = list(range(n))
        for _ in range(self.update_epochs):
            random.shuffle(indices)
            for start in range(0, n, self.batch_size):
                idx = indices[start:start + self.batch_size]
                batch_s = states[idx]
                batch_a = actions[idx]
                batch_olp = old_log_probs[idx]
                batch_ret = returns[idx]
                batch_adv = advantages[idx]

                logits, values = self.net(batch_s)
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(batch_a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_olp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values.squeeze(), batch_ret)
                loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                total_loss += loss.item()
                self.train_steps += 1

        return {'loss': total_loss / max(len(indices) // self.batch_size, 1), 'mean_return': float(returns.mean().item())}

    def end_episode(self):
        if not self._trajectory:
            return
        result = self.update(self._trajectory)
        self._trajectory.clear()

    def save(self, model_id: str) -> str:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        torch.save({
            'algorithm': self.algorithm,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'lr': self.lr,
            'gamma': self.gamma,
        }, path)
        return str(path)

    def load(self, model_id: str) -> bool:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.input_dim = ckpt['input_dim']
        self.hidden_dim = ckpt['hidden_dim']
        self.output_dim = ckpt['output_dim']
        self.lr = ckpt.get('lr', self.lr)
        self.gamma = ckpt.get('gamma', self.gamma)
        self.net = PPONetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_steps = ckpt.get('train_steps', 0)
        return True


# ═══════════════════════════════════════════════════════════════
#  A2C
# ═══════════════════════════════════════════════════════════════

class A2CNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 3):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, output_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.fc(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value


class A2CAgent:
    algorithm = "a2c"

    def __init__(
        self,
        input_dim: int = 12,
        hidden_dim: int = 128,
        output_dim: int = 3,
        lr: float = 7e-4,
        gamma: float = 0.99,
        n_steps: int = 5,
        entropy_coef: float = 0.01,
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.lr = lr
        self.gamma = gamma
        self.n_steps = n_steps
        self.entropy_coef = entropy_coef
        self.device = DEVICE
        self.train_steps = 0

        self.net = A2CNetwork(input_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)

        self._trajectory: List[Dict] = []

    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        logits, value = self.net(s)
        probs = F.softmax(logits, dim=-1)
        if eval_mode:
            return int(probs.argmax().item())
        dist = torch.distributions.Categorical(probs)
        a = dist.sample()
        log_prob = dist.log_prob(a)
        self._trajectory.append({
            'state': state,
            'action': int(a.item()),
            'log_prob': float(log_prob.item()),
            'value': float(value.item()),
        })
        return int(a.item())

    def train_step(self, reward: float, next_state: np.ndarray, done: bool) -> Optional[Dict[str, float]]:
        if not self._trajectory:
            return None
        self._trajectory[-1]['reward'] = reward
        self._trajectory[-1]['next_state'] = next_state
        self._trajectory[-1]['done'] = done
        if len(self._trajectory) < self.n_steps and not done:
            return None
        return self._update()

    def _update(self) -> Dict[str, float]:
        traj = self._trajectory
        T = len(traj)

        states = torch.tensor(np.array([t['state'] for t in traj]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([t['action'] for t in traj], dtype=torch.long, device=self.device)
        rewards = torch.tensor([t['reward'] for t in traj], dtype=torch.float32, device=self.device)
        old_values = torch.tensor([t['value'] for t in traj], dtype=torch.float32, device=self.device)

        returns = []
        g = 0.0
        with torch.no_grad():
            final_s = torch.tensor(traj[-1]['next_state'], dtype=torch.float32, device=self.device).unsqueeze(0)
            _, final_val_t = self.net(final_s)
            final_val = 0.0 if traj[-1]['done'] else final_val_t.item()
        for i in reversed(range(T)):
            g = rewards[i].item() + self.gamma * g
            returns.insert(0, g)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)

        advantages = returns - old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        logits, values = self.net(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        new_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        policy_loss = -(new_log_probs * advantages).mean()
        value_loss = F.mse_loss(values.squeeze(), returns)
        loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
        self.optimizer.step()
        self.train_steps += 1

        self._trajectory.clear()
        return {'loss': float(loss.item()), 'mean_return': float(returns.mean().item())}

    def end_episode(self):
        if self._trajectory:
            return self._update()
        return None

    def save(self, model_id: str) -> str:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        torch.save({
            'algorithm': self.algorithm,
            'state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_steps': self.train_steps,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'output_dim': self.output_dim,
            'lr': self.lr,
            'gamma': self.gamma,
        }, path)
        return str(path)

    def load(self, model_id: str) -> bool:
        path = WEIGHTS_DIR / f"{model_id}.pt"
        if not path.exists():
            return False
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        self.input_dim = ckpt['input_dim']
        self.hidden_dim = ckpt['hidden_dim']
        self.output_dim = ckpt['output_dim']
        self.lr = ckpt.get('lr', self.lr)
        self.gamma = ckpt.get('gamma', self.gamma)
        self.net = A2CNetwork(self.input_dim, self.hidden_dim, self.output_dim).to(self.device)
        self.net.load_state_dict(ckpt['state_dict'])
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.train_steps = ckpt.get('train_steps', 0)
        return True


# ═══════════════════════════════════════════════════════════════
#  Factory
# ═══════════════════════════════════════════════════════════════

def _ppo_continuous_factory(**kwargs):
    """Lazy import to avoid a circular dependency at module load time."""
    from .ppo_continuous import PPOContinuousAgent
    # Filter to args PPOContinuousAgent understands. The /rl/train route passes
    # generic params like hidden_dim/lr/learning_rate that should be respected,
    # but DQN-only params (epsilon, buffer_capacity) must be silently dropped.
    accepted = {"state_dim", "hidden_dim", "lstm_layers", "window", "lr",
                "gamma", "lam", "clip_epsilon", "value_clip", "entropy_coef",
                "value_coef", "max_grad_norm", "update_epochs", "batch_size",
                "rollout_steps"}
    if "learning_rate" in kwargs and "lr" not in kwargs:
        kwargs["lr"] = kwargs.pop("learning_rate")
    if "input_dim" in kwargs and "state_dim" not in kwargs:
        # Legacy callers may still pass input_dim=12; ignore it — PPO-continuous
        # uses STATE_DIM (30) internally.
        kwargs.pop("input_dim")
    return PPOContinuousAgent(**{k: v for k, v in kwargs.items() if k in accepted})


ALGORITHMS = {
    'dqn': DQNAgent,
    'ppo': PPOAgent,
    'a2c': A2CAgent,
    'ppo_continuous': _ppo_continuous_factory,
}


def create_agent(algorithm: str = 'dqn', **kwargs):
    factory = ALGORITHMS.get(algorithm)
    if factory is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from: {list(ALGORITHMS.keys())}")
    return factory(**kwargs)


def list_models() -> List[Dict[str, Any]]:
    models = []
    for f in sorted(WEIGHTS_DIR.glob("*.pt"), reverse=True):
        try:
            ckpt = torch.load(f, map_location='cpu', weights_only=True)
            alg = ckpt.get('algorithm', 'dqn')
            models.append({
                'id': f.stem,
                'path': str(f),
                'algorithm': alg,
                'created': f.stem.split('_')[-1] if '_' in f.stem else '',
                'input_dim': ckpt.get('input_dim', 0),
                'hidden_dim': ckpt.get('hidden_dim', 0),
                'train_steps': ckpt.get('train_steps', 0),
                'epsilon': round(ckpt.get('epsilon', 0), 4),
            })
        except Exception:
            models.append({'id': f.stem, 'path': str(f), 'error': 'corrupt'})
    return models


def load_agent(model_id: str) -> Any:
    path = WEIGHTS_DIR / f"{model_id}.pt"
    if not path.exists():
        return None
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    alg = ckpt.get('algorithm', 'dqn')
    agent = create_agent(alg)
    if not agent.load(model_id):
        return None
    return agent
