"""
Training script for PPO agent on multi-market energy trading environment.

This script demonstrates how to:
1. Configure which markets to include
2. Train a PPO agent
3. Evaluate the trained agent
4. Save and load models
"""

import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

from base_env import BaseEnvironment


from nrgise import EnergySystem, Grid, StorageSystemEnergyReservoir


STORAGE_POWER = 200.0  # kWh
STORAGE_CAPACITY = 1000.0  # kW
START_DATE = pd.Timestamp("2024-01-01 00:00:00")
TIME_DELTA_SECONDS = 900  # 15 minutes
MAX_EPISODE_STEPS = 96 * 7  # 7 days with 15 min steps

def create_env(
    energy_system: EnergySystem,
    forecast_horizon_hours: int = 3,

    time_delta_seconds: int = 900,
    battery_power_kwh: float = 50.0,
    battery_capacity_kwh: float = 100.0,
    max_episode_steps: int = 96 * 7,
    start_date: pd.Timestamp = START_DATE,
) -> BaseEnvironment:
    """Create and validate environment."""
    env = BaseEnvironment(
        # forecaster=None,  # Placeholder, can be used for historical data if needed
        energy_system=energy_system,
        forecast_horizon_hours=forecast_horizon_hours,
        time_delta_seconds = time_delta_seconds,
        battery_power_kwh=battery_power_kwh,
        battery_capacity_kwh=battery_capacity_kwh,
        max_episode_steps=max_episode_steps,  # 24 hours with 15 min steps
        start_date=start_date,
    )
    
    return env


def train_ppo_agent(
    env: BaseEnvironment,
    total_timesteps: int = 100_000,
    save_path: str = './models',
    model_name: str = 'ppo_energy_trading',
    eval_freq: int = 5000,
    checkpoint_freq: int = 10000,
    use_vec_normalize: bool = True,
    verbose: int = 1
) -> PPO:
    """
    Train a PPO agent on the energy trading environment.
    
    Parameters:
    -----------
    env : EnergyTradingEnv
        The trading environment
    total_timesteps : int
        Total number of training steps
    save_path : str
        Directory to save models and logs
    model_name : str
        Name for the saved model
    eval_freq : int
        Frequency of evaluation (in timesteps)
    checkpoint_freq : int
        Frequency of saving checkpoints
    use_vec_normalize : bool
        Whether to normalize observations and rewards
    verbose : int
        Verbosity level
        
    Returns:
    --------
    model : PPO
        Trained PPO model
    """
    
    # Create directories
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(f'{save_path}/checkpoints', exist_ok=True)
    os.makedirs(f'{save_path}/logs', exist_ok=True)
    
    # Wrap in vectorized environment
    vec_env = DummyVecEnv([lambda: env])
    
    # Optionally normalize observations and rewards
    if use_vec_normalize:
        vec_env = VecNormalize(
            vec_env,
            norm_obs=True,
            norm_reward=True,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: create_env(
        energy_system=env.energy_system,
        forecast_horizon_hours=env.forecast_horizon_hours,
        time_delta_seconds=env.time_delta_seconds,
        battery_power_kwh=env.battery_power_kwh,
        battery_capacity_kwh=env.battery_capacity_kwh,
        max_episode_steps=env.max_episode_steps,
        start_date=env.start_date,
    )])
    if use_vec_normalize:
        eval_env = VecNormalize(
            eval_env,
            norm_obs=True,
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0,
            training=False
        )
    
    # Callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f'{save_path}/best',
        log_path=f'{save_path}/logs',
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        verbose=verbose
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=f'{save_path}/checkpoints',
        name_prefix=model_name,
        save_vecnormalize=use_vec_normalize
    )
    
    # Create PPO model
    model = PPO(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        n_steps=97 * 7,
        batch_size=64 * 4,  # Increase batch size for better performance
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        tensorboard_log=f'{save_path}/logs'
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    # trading_callback = TradingCallback(
    #     log_freq=96,
    #     battery_capacity_kwh=STORAGE_CAPACITY,
    #     verbose=1,
    # )
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback,],
        progress_bar=True,
        tb_log_name="ppo_run",
    )
    
    # Save final model
    model.save(f'{save_path}/{model_name}_final')
    if use_vec_normalize:
        vec_env.save(f'{save_path}/{model_name}_vecnormalize.pkl')
    
    print(f"Training completed! Model saved to {save_path}/{model_name}_final")
    
    return model


class InfoLoggerCallback(BaseCallback):
    """
    Logs all fields from the info dict to TensorBoard/stdout during training.
    Access via: tensorboard --logdir ./logs
    """

    def __init__(self, log_freq: int = 100, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

        # Rolling buffers for aggregation
        self._idc_rewards    = []
        self._energy_rewards = []

    def _on_step(self) -> bool:
        # self.locals["infos"] is a list — one entry per parallel env
        for info in self.locals["infos"]:

            # ── Pull whatever keys your step() puts in info ───────────────
            idc_r    = info.get("idc_reward",    0.0)
            energy_r = info.get("energy_reward", 0.0)

            self._idc_rewards.append(idc_r)
            self._energy_rewards.append(energy_r)

        # Log aggregated stats every log_freq steps
        if self.num_timesteps % self.log_freq == 0:
            self.logger.record("custom/idc_reward_mean",
                               float(np.mean(self._idc_rewards[-self.log_freq:])))
            self.logger.record("custom/energy_penalty_mean",
                               float(np.mean(self._energy_rewards[-self.log_freq:])))
            self.logger.record("custom/energy_penalty_sum",
                               float(np.sum(self._energy_rewards[-self.log_freq:])))

            if self.verbose > 0:
                print(f"[step {self.num_timesteps}] "
                      f"idc={np.mean(self._idc_rewards[-self.log_freq:]):.2f}  "
                      f"penalty={np.sum(self._energy_rewards[-self.log_freq:]):.2f}")

        return True   # returning False would stop training early

def evaluate_agent(
    model: PPO,
    env: BaseEnvironment,
    n_episodes: int = 100,
    render: bool = False
) -> Dict[str, List[float]]:
    episode_returns = []      # sum of rewards over the episode
    episode_mean_rewards = [] # per-step average reward
    episode_socs = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_return = 0.0
        n_steps = 0
        done = False
        socs = [info['initial_soc']]

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_return += reward
            n_steps += 1
            socs.append(info['soc'])

            if render:
                env.render()

        mean_reward = episode_return / n_steps if n_steps > 0 else 0.0
        episode_returns.append(episode_return)
        episode_mean_rewards.append(mean_reward)
        # episode_revenues.append(info['total_revenue'])
        episode_socs.append(socs)

        # print(
        #     f"Episode {episode + 1}/{n_episodes}: "
        #     f"Return={episode_return:.2f}  "
        #     f"MeanReward/step={mean_reward:.4f}  "
        #     f"Steps={n_steps}  "
        #     f"Revenue=€{info['total_revenue']:.2f}"
        # )

    results = {
        'returns': episode_returns,           # primary metric
        'mean_rewards': episode_mean_rewards, # per-step average for comparison
        # 'revenues': episode_revenues,
        'socs': episode_socs,
        # episode return stats
        'mean_return': np.mean(episode_returns),
        'std_return': np.std(episode_returns),
        # per-step mean reward stats
        'mean_reward': np.mean(episode_mean_rewards),
        'std_reward': np.std(episode_mean_rewards),
        # revenue stats
        # 'mean_revenue': np.mean(episode_revenues),
        # 'std_revenue': np.std(episode_revenues),
    }

    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"  Episode Return  : {results['mean_return']:.2f} ± {results['std_return']:.2f}")
    print(f"  Mean Reward/step: {results['mean_reward']:.4f} ± {results['std_reward']:.4f}")
    # print(f"  Mean Revenue    : €{results['mean_revenue']:.2f} ± €{results['std_revenue']:.2f}")

    return results

def plot_results(results: Dict, save_path: Optional[str] = None):
    """Plot evaluation results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # ── Top-left: Episode Return with rolling mean ──────────────────────────
    ax1 = axes[0, 0]
    returns = results['returns']
    ax1.plot(returns, marker='o', color='steelblue', alpha=0.6, label='Episode Return')
    ax1.axhline(y=results['mean_return'], color='blue', linestyle='--', label=f"Mean ({results['mean_return']:.1f})")
    window = max(1, len(returns) // 5)
    if len(returns) >= window:
        rolling = np.convolve(returns, np.ones(window) / window, mode='valid')
        ax1.plot(range(window - 1, len(returns)), rolling, color='red', linewidth=2, label=f'Rolling mean (w={window})')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Return')
    ax1.set_title('Episode Return over Evaluation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Top-right: Return distribution ─────────────────────────────────────
    ax2 = axes[0, 1]
    ax2.hist(returns, bins=max(5, len(returns) // 3), color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=results['mean_return'], color='blue', linestyle='--',
                label=f"Mean: {results['mean_return']:.1f} ± {results['std_return']:.1f}")
    ax2.set_xlabel('Episode Return')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Return Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # ── Bottom-left: SOC evolution (first episode) ──────────────────────────
    if results['socs']:
        axes[1, 0].plot(results['socs'][0], color='orange')
        axes[1, 0].set_xlabel('Time Step')
        axes[1, 0].set_ylabel('State of Charge')
        axes[1, 0].set_title('SOC Evolution (First Episode)')
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].grid(True, alpha=0.3)

    # ── Bottom-right: SOC evolution (all episodes, faded) ───────────────────
    ax4 = axes[1, 1]
    for soc_trace in results['socs']:
        ax4.plot(soc_trace, color='orange', alpha=0.3, linewidth=0.8)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('State of Charge')
    ax4.set_title('SOC Evolution (All Episodes)')
    ax4.set_ylim([0, 1])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.show()


def generate_prices(base_price: float, price_volatility: float, max_steps: int) -> np.ndarray:
        """Generate synthetic energy prices with daily and weekly patterns."""
        np.random.seed(42)
        
        t = np.arange(max_steps)
        
        # Daily pattern (peak during day, low at night)
        daily_pattern = 10 * np.sin(2 * np.pi * t / 96 - np.pi / 2)
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 5 * (1 - np.cos(2 * np.pi * t / (96 * 7)))
        
        # Random walk component
        random_walk = np.cumsum(np.random.randn(max_steps) * price_volatility)
        
        # Combine components
        prices = base_price + daily_pattern + weekly_pattern + random_walk
        prices = np.maximum(prices, 10.0)  # Floor price
        
        
        return prices

def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("Multi-Market Energy Trading - PPO Training")
    print("=" * 60)
    
    # Generate sample price profiles
    print("\n1. Generating price profiles...")
    # price_profiles = create_sample_price_profiles(days=30, time_delta_seconds=900)
    timestamps = pd.date_range(start=START_DATE, periods=MAX_EPISODE_STEPS, freq='15T')
    
    idc_price_profile = generate_prices(base_price=80, price_volatility=0.05, max_steps=MAX_EPISODE_STEPS)
    idc_price_profile = idc_price_profile / 1000.  # Convert to €/kWh
    # Configure energy system
    print("\n2. Configuring energy system...")

    energy_system = EnergySystem(time_index=timestamps)
    grid = Grid(label='grid')

    battery = StorageSystemEnergyReservoir(label ='battery',
                                           time_delta_seconds=900,
                                           nom_power=STORAGE_POWER,
                                           capacity=STORAGE_CAPACITY,
                                           initial_soc=0.5,
                                           eta_charge=1,
                                           eta_discharge=1,
                                           )
    
    energy_system.add_components(grid, battery)
   
    
    
    
    # Create environment
    print("\n3. Creating environment...")
    env = create_env(
        energy_system=energy_system,
        forecast_horizon_hours=3,
        time_delta_seconds = TIME_DELTA_SECONDS,
        battery_power_kwh=STORAGE_POWER,
        battery_capacity_kwh=STORAGE_CAPACITY,
        max_episode_steps=MAX_EPISODE_STEPS,  # 24 hours with 15 min steps
        start_date=START_DATE,
    )
    
    # Check environment
    print("\n4. Validating environment...")
    check_env(env, warn=True)
    print("   Environment validation passed!")
    
    obs, info = env.reset()
    print(f"   Initial observation shape: {obs.shape}")
    print(f"   Initial info: {info}")

    # Train agent
    print("\n5. Training PPO agent...")
    model = train_ppo_agent(
        env=env,
        total_timesteps=12_000,  # Increase for better results
        save_path='./models/base_env',
        model_name='ppo_trading',
        eval_freq=1_000,
        checkpoint_freq=5_000,
        use_vec_normalize=True,
        verbose=0
    )
    
    # Evaluate agent
    print("\n6. Evaluating trained agent...")
    results = evaluate_agent(
        model=model,
        env=env,
        n_episodes=10,
        render=False
    )
    
    # Plot results
    print("\n7. Plotting results...")
    plot_results(results, save_path='./models/base_env/evaluation_results.png')
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)


if __name__ == '__main__':
    main()
