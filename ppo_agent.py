import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
from typing import Tuple, List
from collections import deque


class ActorCritic(nn.Module):
    """Actor-Critic network for PPO"""
    
    def __init__(self, 
                 obs_dim: int, 
                 action_dim: int, 
                 hidden_dims: List[int] = [256, 256]):
        super(ActorCritic, self).__init__()
        
        # Shared feature extraction
        layers = []
        prev_dim = obs_dim
        for hidden_dim in hidden_dims[:-1]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)
            ])
            prev_dim = hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )
        
        # Log standard deviation (learnable)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            distribution: Normal distribution for actions
            value: State value estimate
        """
        features = self.shared_net(x)
        
        # Actor
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        distribution = Normal(action_mean, action_std)
        
        # Critic
        value = self.critic(features)
        
        return distribution, value
    
    def get_action(self, x: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy
        
        Returns:
            action: Sampled action
            log_prob: Log probability of action
            value: State value estimate
        """
        distribution, value = self.forward(x)
        
        if deterministic:
            action = distribution.mean
        else:
            action = distribution.sample()
        
        log_prob = distribution.log_prob(action).sum(dim=-1, keepdim=True)
        
        return action, log_prob, value
    
    def evaluate_actions(self, x: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate actions (for training)
        
        Returns:
            log_prob: Log probability of actions
            value: State value estimate
            entropy: Policy entropy
        """
        distribution, value = self.forward(x)
        
        log_prob = distribution.log_prob(actions).sum(dim=-1, keepdim=True)
        entropy = distribution.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, value, entropy


class RolloutBuffer:
    """Buffer for storing rollout data"""
    
    def __init__(self):
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, obs, action, log_prob, reward, value, done):
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def get(self):
        """Get all data as tensors"""
        return {
            'observations': torch.FloatTensor(np.array(self.observations)),
            'actions': torch.FloatTensor(np.array(self.actions)),
            'log_probs': torch.FloatTensor(np.array(self.log_probs)),
            'rewards': torch.FloatTensor(np.array(self.rewards)),
            'values': torch.FloatTensor(np.array(self.values)),
            'dones': torch.FloatTensor(np.array(self.dones))
        }
    
    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()
    
    def __len__(self):
        return len(self.observations)


class PPO:
    """Proximal Policy Optimization agent"""
    
    def __init__(self,
                 obs_dim: int,
                 action_dim: int,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95,
                 clip_epsilon: float = 0.2,
                 value_loss_coef: float = 0.5,
                 entropy_coef: float = 0.01,
                 max_grad_norm: float = 0.5,
                 n_epochs: int = 10,
                 batch_size: int = 64,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize PPO agent
        
        Args:
            obs_dim: Observation space dimension
            action_dim: Action space dimension
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            clip_epsilon: PPO clipping parameter
            value_loss_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            max_grad_norm: Maximum gradient norm for clipping
            n_epochs: Number of epochs per update
            batch_size: Batch size for training
            device: Device to use (cpu/cuda)
        """
        self.device = torch.device(device)
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Networks
        self.policy = ActorCritic(obs_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Rollout buffer
        self.buffer = RolloutBuffer()
        
        # Tracking
        self.update_count = 0
    
    def select_action(self, observation: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation"""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            action, log_prob, value = self.policy.get_action(obs_tensor, deterministic)
            
        return action.cpu().numpy()[0], log_prob.cpu().numpy()[0], value.cpu().numpy()[0][0]
    
    def store_transition(self, obs, action, log_prob, reward, value, done):
        """Store transition in buffer"""
        self.buffer.add(obs, action, log_prob, reward, value, done)
    
    def compute_gae(self, rewards, values, dones, next_value):
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        values = torch.cat([values, next_value.unsqueeze(0)])
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value_t = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value_t * next_non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
            advantages[t] = gae
        
        returns = advantages + values[:-1]
        
        return advantages, returns
    
    def update(self, next_obs: np.ndarray):
        """Update policy using collected rollouts"""
        if len(self.buffer) == 0:
            return {}
        
        # Get data from buffer
        data = self.buffer.get()
        observations = data['observations'].to(self.device)
        actions = data['actions'].to(self.device)
        old_log_probs = data['log_probs'].to(self.device)
        rewards = data['rewards'].to(self.device)
        values = data['values'].to(self.device)
        dones = data['dones'].to(self.device)
        
        # Compute next value
        with torch.no_grad():
            next_obs_tensor = torch.FloatTensor(next_obs).unsqueeze(0).to(self.device)
            _, next_value = self.policy.forward(next_obs_tensor)
            next_value = next_value.squeeze()
        
        # Compute advantages and returns
        advantages, returns = self.compute_gae(rewards, values, dones, next_value)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Training loop
        dataset_size = len(observations)
        indices = np.arange(dataset_size)
        
        policy_losses = []
        value_losses = []
        entropies = []
        
        for _ in range(self.n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]
                
                # Get batch data
                batch_obs = observations[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # Evaluate actions
                log_probs, values, entropy = self.policy.evaluate_actions(batch_obs, batch_actions)
                
                # Policy loss (PPO clipped objective)
                ratio = torch.exp(log_probs - batch_old_log_probs)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss
                value_loss = nn.MSELoss()(values.squeeze(), batch_returns)
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # Track losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())
        
        # Clear buffer
        self.buffer.clear()
        self.update_count += 1
        
        # Return training statistics
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy': np.mean(entropies),
            'update_count': self.update_count
        }
    
    def save(self, path: str):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.update_count = checkpoint['update_count']
