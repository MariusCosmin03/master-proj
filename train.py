"""
Training script for PPO agent on multi-market energy trading environment.

This script demonstrates how to:
1. Configure which markets to include
2. Train a PPO agent
3. Evaluate the trained agent
4. Save and load models
"""

from datetime import datetime
from importlib.resources import as_file

import numpy as np
import pandas as pd
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os

from environment import TradingEnvironment
from markets.daa_market import DaaMarket
from markets.idc_market import IdcMarket
from markets_wrapper import MarketsWrapper

from nrgise import EnergySystem, Grid, StorageSystemEnergyReservoir
from nrgise.forecaster import DataProfileForecaster

from training_callback import TradingCallback


STORAGE_POWER = 1000.0  # kWh
STORAGE_CAPACITY = 1000.0  # kW
START_DATE = pd.Timestamp("2024-01-01 00:00:00")
TIME_DELTA_SECONDS = 900  # 15 minutes
MAX_EPISODE_STEPS = 96 * 7  # 7 days with 15 min steps

def create_env(
    energy_system: EnergySystem,
    markets: MarketsWrapper,
    forecast_horizon_hours: int = 3,
    idc_price_data=None,
    daa_price_data=None,
    time_delta_seconds: int = 900,
    battery_power_kwh: float = 50.0,
    battery_capacity_kwh: float = 100.0,
    max_episode_steps: int = 96 * 20,
    start_date: pd.Timestamp = START_DATE,
) -> TradingEnvironment:
    """Create and validate environment."""
    env = TradingEnvironment(
        # forecaster=None,  # Placeholder, can be used for historical data if needed
        energy_system=energy_system,
        markets=markets,
        forecast_horizon_hours=forecast_horizon_hours,
        idc_price_data=idc_price_data,
        daa_price_data=daa_price_data,
        time_delta_seconds = time_delta_seconds,
        battery_power_kwh=battery_power_kwh,
        battery_capacity_kwh=battery_capacity_kwh,
        max_episode_steps=max_episode_steps,  # 24 hours with 15 min steps
        start_date=start_date,
    )
    
    return env


def train_ppo_agent(
    env: TradingEnvironment,
    eval_env: Optional[TradingEnvironment] = None,
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
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Create evaluation environment
    if eval_env is None:
        eval_env = DummyVecEnv([lambda: create_env(
            energy_system=env.energy_system,
            markets=env.markets,
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
        n_steps=2048,
        batch_size=2048,  # Increase batch size for better performance
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=verbose,
        tensorboard_log=f'{save_path}/logs',
        device='cpu',
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    trading_callback = TradingCallback(
        log_freq=96,
        battery_capacity_kwh=STORAGE_CAPACITY,
        verbose=verbose,
    )
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, trading_callback],
        progress_bar=True,
        tb_log_name="ppo_run",
    )
    
    # Save final model
    model.save(f'{save_path}/{model_name}_final')
    if use_vec_normalize:
        vec_env.save(f'{save_path}/{model_name}_vecnormalize.pkl')
    
    print(f"Training completed! Model saved to {save_path}/{model_name}_final")
    
    return model


def train_sac_agent(
    env: TradingEnvironment,
    eval_env: Optional[TradingEnvironment] = None,
    total_timesteps: int = 100_000,
    save_path: str = './models',
    model_name: str = 'sac_energy_trading',
    eval_freq: int = 5000,
    checkpoint_freq: int = 10000,
    use_vec_normalize: bool = True,
    verbose: int = 1
) -> SAC:
    """
    Train a SAC agent on the energy trading environment.
    
    Parameters:
    -----------
    env : TradingEnvironment
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
    model : SAC
        Trained SAC model
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
            norm_reward=False,
            clip_obs=10.0,
            clip_reward=10.0
        )
    
    # Create evaluation environment
    if eval_env is None:
        raise ValueError("eval_env must be provided for SAC training.")
        eval_env = DummyVecEnv([lambda: create_env(
            energy_system=env.energy_system,
            markets=env.markets,
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
            training=False,
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
        save_replay_buffer=True,      # SAC: persist the replay buffer in checkpoints
        save_vecnormalize=use_vec_normalize
    )
    
    # Create SAC model
    # SAC is off-policy: it learns from a replay buffer rather than on-policy rollouts
    model = SAC(
        'MlpPolicy',
        vec_env,
        learning_rate=3e-4,
        buffer_size=1_000_000,        # replay buffer capacity
        learning_starts=10_000,       # steps of random exploration before first update
        batch_size=256,               # mini-batch size drawn from the replay buffer
        tau=0.005,                    # soft target-network update coefficient
        gamma=0.99,
        train_freq=1,                 # update the model every N environment steps
        gradient_steps=1,            # gradient updates per environment step
        ent_coef='auto',              # automatic entropy tuning (SAC hallmark)
        target_update_interval=1,    # target network update interval (steps)
        target_entropy='auto',        # target entropy for automatic tuning
        use_sde=False,                # set True to use State-Dependent Exploration
        sde_sample_freq=-1,
        policy_kwargs=dict(net_arch=[256, 256]),
        verbose=verbose,
        tensorboard_log=f'{save_path}/logs',
        device='cpu',
    )
    
    print(f"Starting training for {total_timesteps} timesteps...")
    
    trading_callback = TradingCallback(
        log_freq=96,
        battery_capacity_kwh=STORAGE_CAPACITY,
        verbose=verbose,
    )
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, trading_callback],
        progress_bar=True,
        tb_log_name="sac_run",
    )
    
    # Save final model
    model.save(f'{save_path}/{model_name}_final')
    if use_vec_normalize:
        vec_env.save(f'{save_path}/{model_name}_vecnormalize.pkl')
    
    print(f"Training completed! Model saved to {save_path}/{model_name}_final")
    
    return model
 

# def make_markets(idc_prices: np.ndarray, daa_prices: np.ndarray) -> MarketsWrapper:
#     """
#     Rebuild all market objects and forecasters from raw price arrays.
 
#     Parameters
#     ----------
#     idc_prices : np.ndarray  shape (steps,)  — 15-min IDC prices
#     daa_prices : np.ndarray  shape (hours,)  — hourly DAA prices
#     """
#     steps = len(idc_prices)
#     idc_timestamps = pd.date_range(start=START_DATE, periods=steps, freq='15min')
#     daa_timestamps = pd.date_range(start=START_DATE, periods=len(daa_prices), freq='h')
 
#     idc_market    = IdcMarket(price_profile_per_simulation_time_step=pd.Series(idc_prices, index=idc_timestamps))
#     idc_forecaster = DataProfileForecaster(idc_prices, time_delta_seconds=900)
 
#     daa_market    = DaaMarket(price_profile_per_simulation_time_step=pd.Series(daa_prices, index=daa_timestamps))
#     daa_forecaster = DataProfileForecaster(np.array(daa_prices), time_delta_seconds=3600)
 
#     return MarketsWrapper(
#         battery_capacity_kwh=STORAGE_CAPACITY,
#         battery_max_power_kwh=STORAGE_POWER,
#         intraday_market=idc_market,
#         idc_price_forcaster=idc_forecaster,
#         day_ahead_market=daa_market,
#         daa_price_forecaster=daa_forecaster,
#         max_steps=steps,
#     )
 
 
# def build_all_envs(
#     idc_weeks: np.ndarray,
#     daa_weeks: np.ndarray,
#     energy_system: EnergySystem,
#     use_vec_normalize: bool,
#     training: bool = True,
# ) -> list[VecNormalize | DummyVecEnv]:
#     """
#     Pre-build one VecEnv per week upfront. Returns a list indexed by week.
 
#     Parameters
#     ----------
#     idc_weeks       : shape (n_weeks, steps_per_week)
#     daa_weeks       : shape (n_weeks, hours_per_week)
#     energy_system   : shared EnergySystem template
#     use_vec_normalize : wrap each env in VecNormalize
#     training        : passed to VecNormalize — False freezes running stats (for eval/test)
#     """
#     envs = []
#     n = len(idc_weeks)
#     for i, (idc, daa) in enumerate(zip(idc_weeks, daa_weeks)):
#         print(f"  Building env {i+1}/{n}...", end='\r')
#         steps = len(idc)
#         markets = make_markets(idc, daa)
#         env = create_env(
#             energy_system=energy_system,
#             markets=markets,
#             forecast_horizon_hours=3,
#             time_delta_seconds=TIME_DELTA_SECONDS,
#             battery_power_kwh=STORAGE_POWER,
#             battery_capacity_kwh=STORAGE_CAPACITY,
#             max_episode_steps=steps,
#             start_date=START_DATE,
#         )
#         vec_env = DummyVecEnv([lambda e=env: e])    # capture env in default arg
#         if use_vec_normalize:
#             vec_env = VecNormalize(
#                 vec_env,
#                 norm_obs=True,
#                 norm_reward=False,
#                 clip_obs=10.0,
#                 clip_reward=10.0,
#                 training=training,
#             )
#         envs.append(vec_env)
#     print(f"  Built {n} environments.")
#     return envs
 

# class WeeklyEvalCallback(BaseCallback):
#     """
#     Evaluates the model on each validation week sequentially (n_envs=1 each),
#     avoiding the DummyVecEnv auto-reset bug that truncates episodes early when
#     multiple sub-envs finish at different timesteps.
 
#     Saves the best model based on mean reward across all val weeks.
#     """
 
#     def __init__(
#         self,
#         val_envs: list,
#         best_model_save_path: str,
#         log_path: str,
#         eval_freq: int = 5000,
#         deterministic: bool = True,
#         verbose: int = 1,
#     ):
#         super().__init__(verbose)
#         self.val_envs             = val_envs
#         self.best_model_save_path = best_model_save_path
#         self.log_path             = log_path
#         self.eval_freq            = eval_freq
#         self.deterministic        = deterministic
#         self.best_mean_reward     = -np.inf
#         os.makedirs(best_model_save_path, exist_ok=True)
#         os.makedirs(log_path, exist_ok=True)
 
#     def _on_step(self) -> bool:
#         if self.n_calls % self.eval_freq != 0:
#             return True
 
#         rewards = []
#         for env in self.val_envs:
#             obs = env.reset()
#             done = False
#             episode_reward = 0.0
#             while not done:
#                 action, _ = self.model.predict(obs, deterministic=self.deterministic)
#                 obs, reward, done, _ = env.step(action)
#                 episode_reward += reward.item()
#             rewards.append(episode_reward)
 
#         mean_reward = float(np.mean(rewards))
#         std_reward  = float(np.std(rewards))
 
#         # Log to TensorBoard
#         self.logger.record("eval/mean_reward", mean_reward)
#         self.logger.record("eval/std_reward",  std_reward)
 
#         if self.verbose:
#             print(f"[WeeklyEvalCallback] step={self.num_timesteps} | "
#                   f"mean_reward={mean_reward:.4f} ± {std_reward:.4f} "
#                   f"over {len(self.val_envs)} val weeks")
 
#         # Save best model
#         if mean_reward > self.best_mean_reward:
#             self.best_mean_reward = mean_reward
#             self.model.save(os.path.join(self.best_model_save_path, "best_model"))
#             if self.verbose:
#                 print(f"  → New best model saved ({mean_reward:.4f})")
 
#         return True
 
 
# def train_sac_agent(
#     energy_system: EnergySystem,
#     train_idc: np.ndarray,          # shape: (n_train_weeks, steps_per_week)
#     train_daa: np.ndarray,          # shape: (n_train_weeks, hours_per_week)
#     val_idc: np.ndarray,            # shape: (n_val_weeks,   steps_per_week)
#     val_daa: np.ndarray,            # shape: (n_val_weeks,   hours_per_week)
#     test_idc: np.ndarray,           # shape: (n_test_weeks,  steps_per_week)
#     test_daa: np.ndarray,           # shape: (n_test_weeks,  hours_per_week)
#     n_epochs: int = 5,
#     save_path: str = './models',
#     model_name: str = 'sac_energy_trading',
#     eval_freq: int = 5000,
#     checkpoint_freq: int = 10000,
#     use_vec_normalize: bool = True,
#     verbose: int = 1,
# ) -> SAC:
#     """
#     Train a SAC agent using weekly IDC + DAA price episodes.
 
#     All environments (train, val, test) are built once before training starts.
#     Each training iteration simply calls model.set_env() with a pre-built env —
#     no market or env construction overhead during the training loop.
#     """
 
#     os.makedirs(save_path, exist_ok=True)
#     os.makedirs(f'{save_path}/checkpoints', exist_ok=True)
#     os.makedirs(f'{save_path}/logs', exist_ok=True)
 
#     steps_per_week = train_idc.shape[1]
 
#     # ── Pre-build all environments once ────────────────────────────────────
#     print("Pre-building training environments...")
#     train_envs = build_all_envs(train_idc, train_daa, energy_system, use_vec_normalize, training=True)
 
#     print("Pre-building validation environments...")
#     val_envs = build_all_envs(val_idc, val_daa, energy_system, use_vec_normalize, training=False)
 
#     print("Pre-building test environments...")
#     test_envs = build_all_envs(test_idc, test_daa, energy_system, use_vec_normalize, training=False)
 

 
#     # ── Callbacks ──────────────────────────────────────────────────────────
#     eval_callback = WeeklyEvalCallback(
#         val_envs=val_envs,
#         best_model_save_path=f'{save_path}/best',
#         log_path=f'{save_path}/logs',
#         eval_freq=eval_freq,
#         deterministic=True,
#         verbose=verbose,
#     )

#     checkpoint_callback = CheckpointCallback(
#         save_freq=checkpoint_freq,
#         save_path=f'{save_path}/checkpoints',
#         name_prefix=model_name,
#         save_replay_buffer=True,
#         save_vecnormalize=use_vec_normalize,
#     )
#     trading_callback = TradingCallback(
#         log_freq=96,
#         battery_capacity_kwh=STORAGE_CAPACITY,
#         verbose=verbose,
#     )
 
#     # ── Model (initialised on week 0) ──────────────────────────────────────
#     model = SAC(
#         'MlpPolicy',
#         train_envs[0],
#         learning_rate=3e-4,
#         buffer_size=1_000_000,
#         learning_starts=10_000,
#         batch_size=256,
#         tau=0.005,
#         gamma=0.99,
#         train_freq=1,
#         gradient_steps=1,
#         ent_coef='auto',
#         target_update_interval=1,
#         target_entropy='auto',
#         use_sde=False,
#         sde_sample_freq=-1,
#         policy_kwargs=dict(net_arch=[256, 256]),
#         verbose=verbose,
#         tensorboard_log=f'{save_path}/logs',
#         device='cpu',
#     )
 
#     # ── Training loop ──────────────────────────────────────────────────────
#     run_name = f"sac_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

#     total_timesteps = n_epochs * len(train_envs) * steps_per_week
#     print(f"\nTraining for {n_epochs} epochs x {len(train_envs)} weeks "
#           f"= {total_timesteps:,} total timesteps")
 
#     for epoch in range(n_epochs):
#         shuffled_idx = np.random.permutation(len(train_envs))
 
#         for i, week_idx in enumerate(shuffled_idx):
#             print(f"[epoch {epoch+1}/{n_epochs} | week {i+1}/{len(train_envs)}]")
 
#             model.set_env(train_envs[week_idx])     # swap — no construction cost
 
#             model.learn(
#                 total_timesteps=steps_per_week,
#                 callback=[eval_callback, checkpoint_callback, trading_callback],
#                 reset_num_timesteps=False,
#                 tb_log_name=run_name,
#                 progress_bar=False,
#             )
 
#     # ── Save ───────────────────────────────────────────────────────────────
#     model.save(f'{save_path}/{model_name}_final')
#     if use_vec_normalize:
#         train_envs[0].save(f'{save_path}/{model_name}_vecnormalize.pkl')
#     print(f"Model saved to '{save_path}/{model_name}_final'")
 
#     # ── Test evaluation (once, best checkpoint, never influences training) ──
#     print("\nEvaluating on held-out test set...")
#     best_model = SAC.load(f'{save_path}/best/best_model')
#     test_rewards = []
 
#     results = evaluate_policy(best_model, test_envs, deterministic=True)
#     test_rewards.append(results['mean_reward'])
 
#     print(f"Test set — mean reward: {np.mean(test_rewards):.4f} "
#           f"± {np.std(test_rewards):.4f}  (over {len(test_rewards)} weeks)")
 
#     return model

# def evaluate_policy(
#     model: SAC,
#     test_envs: list,
#     deterministic: bool = True,
#     verbose: int = 1,
# ) -> dict:
#     """
#     Evaluate the policy on each pre-built test environment (one per week).
 
#     Parameters
#     ----------
#     model       : trained SAC model (or best checkpoint)
#     test_envs   : list of VecEnvs, one per test week (from build_all_envs)
#     deterministic : use deterministic actions
#     verbose     : print per-week results if > 0
 
#     Returns
#     -------
#     results : dict with keys
#         'rewards'       — list of total reward per week
#         'mean_reward'   — mean across weeks
#         'std_reward'    — std across weeks
#         'min_reward'    — worst week
#         'max_reward'    — best week
#     """
#     rewards = []
 
#     for i, env in enumerate(test_envs):
#         obs = env.reset()
#         done = False
#         episode_reward = 0.0
 
#         while not done:
#             action, _ = model.predict(obs, deterministic=deterministic)
#             obs, reward, done, _ = env.step(action)
#             episode_reward += reward.item()
 
#         rewards.append(episode_reward)
 
#         if verbose:
#             print(f"  Week {i+1:02d}/{len(test_envs)} | reward: {episode_reward:.4f}")
 
#     results = {
#         'rewards':     rewards,
#         'mean_reward': float(np.mean(rewards)),
#         'std_reward':  float(np.std(rewards)),
#         'min_reward':  float(np.min(rewards)),
#         'max_reward':  float(np.max(rewards)),
#     }
 
#     print(f"\nTest set results over {len(test_envs)} weeks:")
#     print(f"  Mean : {results['mean_reward']:>10.4f}")
#     print(f"  Std  : {results['std_reward']:>10.4f}")
#     print(f"  Min  : {results['min_reward']:>10.4f}")
#     print(f"  Max  : {results['max_reward']:>10.4f}")
 
#     return results
 

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
    model,
    env: TradingEnvironment,
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


def continue_training(
    model_path: str,
    env,
    total_timesteps: int = 1_000_000,
    save_dir: str = "models/continued",
    checkpoint_freq: int = 50_000,
    eval_env=None,
    reset_num_timesteps: bool = False,  # False = continues step count from checkpoint
):
    """
    Load an existing  model and continue training.

    Args:
        model_path:          Path to saved model (.zip)
        env:                 Training environment instance
        total_timesteps:     Additional timesteps to train for
        save_dir:            Directory to save new checkpoints
        checkpoint_freq:     Save a checkpoint every N steps
        eval_env:            Optional separate env for EvalCallback
        reset_num_timesteps: If True, resets step counter (affects LR schedule)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Load existing model, swap in the new env
    model = PPO.load(
        model_path,
        env=env,
        device="cpu",
    )

    print(f"Loaded model from: {model_path}")
    print(f"Resuming from step: {model.num_timesteps}")

    # Build callbacks
    callbacks = [
        CheckpointCallback(
            save_freq=checkpoint_freq,
            save_path=save_dir,
            name_prefix="ppo_continued",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
    ]

    if eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "best"),
                log_path=os.path.join(save_dir, "eval_logs"),
                eval_freq=checkpoint_freq,
                deterministic=True,
                render=False,
            )
        )

    callbacks.append(TradingCallback(
        log_freq=96,
        battery_capacity_kwh=STORAGE_CAPACITY,
        verbose=0,
    ))

    # Continue training
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=callbacks,
        progress_bar=True,
        tb_log_name="ppo_continued_run",
    )

    # Save final model
    final_path = os.path.join(save_dir, "ppo_final")
    model.save(final_path)
    print(f"Saved final model to: {final_path}")

    return model


def generate_prices(base_price: float, price_volatility: float, max_steps: int, seed: int = 42) -> np.ndarray:
        """Generate synthetic energy prices with daily and weekly patterns."""
        np.random.seed(seed)
        
        t = np.arange(max_steps)
        
        # Daily pattern (peak during day, low at night)
        daily_pattern = 10 * np.sin(2 * np.pi * t / 96 - np.pi / 2)
        
        # Weekly pattern (higher on weekdays)
        weekly_pattern = 5 * (1 - np.cos(2 * np.pi * t / (96 * 7)))
        
        # Random walk component
        random_walk = np.cumsum(np.random.randn(max_steps) * price_volatility)
        
        # Combine components
        prices = base_price + daily_pattern  + weekly_pattern + random_walk
        # prices = np.maximum(prices, 10.0)  # Floor price


        return prices / 1000.  # Convert to €/kWh


def load_daa_data(csv_path: str) -> pd.Series:
    
    day_ahead_prices = pd.read_csv(csv_path, delimiter=',', index_col=0, parse_dates=True)
    day_ahead_prices_hourly = day_ahead_prices.resample('h').mean()
    day_ahead_prices_hourly = day_ahead_prices_hourly/1000  # convert fom €/MWh to €/kWh

    return day_ahead_prices_hourly# .head(MAX_EPISODE_STEPS // 4 + 24)  # 1 day ahead with hourly steps


# def load_and_evaluate(
#     model_path: str,
#     test_idc: np.ndarray,           # shape: (n_test_weeks, steps_per_week)
#     test_daa: np.ndarray,           # shape: (n_test_weeks, hours_per_week)
#     energy_system: EnergySystem,
#     use_vec_normalize: bool = True,
#     verbose: int = 1,
# ) -> dict:
#     """
#     Load a saved SAC model and evaluate it on the test set.
 
#     Parameters
#     ----------
#     model_path      : path to the saved model, e.g. './models/best/best_model'
#                       (.zip extension is optional)
#     test_idc        : weekly IDC price arrays, shape (n_test_weeks, steps_per_week)
#     test_daa        : weekly DAA price arrays, shape (n_test_weeks, hours_per_week)
#     energy_system   : EnergySystem instance (used to build test envs)
#     use_vec_normalize : must match what was used during training
#     verbose         : print per-week results if > 0
 
#     Returns
#     -------
#     results : dict — see evaluate_policy for full schema
#     """
#     print(f"Loading model from '{model_path}'...")
#     model = SAC.load(model_path)
 
#     print("Building test environments...")
#     test_envs = build_all_envs(
#         test_idc, test_daa, energy_system,
#         use_vec_normalize=use_vec_normalize,
#         training=False,
#     )
 
#     return evaluate_policy(model, test_envs, deterministic=True, verbose=verbose)
 

def main():
    """Main training pipeline."""
    
    print("=" * 60)
    print("Multi-Market Energy Trading - PPO Training")
    print("=" * 60)
    
    # Generate sample price profiles
    print("\n1. Generating price profiles...")
    # price_profiles = create_sample_price_profiles(days=30, time_delta_seconds=900)
    timestamps = pd.date_range(start=START_DATE, periods=MAX_EPISODE_STEPS, freq='15T')
    timestamps_daa = pd.date_range(start=START_DATE, periods=MAX_EPISODE_STEPS/4, freq='1H')

    from scripts.data_split import load_split
    idc_train_weeks = load_split('data_splits/idc/manifest.json', 'train')
    idc_val_weeks   = load_split('data_splits/idc/manifest.json', 'val')
    idc_test_weeks  = load_split('data_splits/idc/manifest.json', 'test')
    idc_price_profile = idc_train_weeks[0]  # Use the first training week as the base profile for the IDC market
    
    # idc_price_profile = generate_prices(base_price=80, price_volatility=0.05, max_steps=MAX_EPISODE_STEPS)
    # idc_price_profile_eval = generate_prices(base_price=80, price_volatility=0.05, max_steps=MAX_EPISODE_STEPS, seed=999)
    idc_price_profile_eval = idc_val_weeks[0]  
    
    daa_train_weeks: list[pd.Series] = load_split("data_splits/daa/manifest.json", "train")  # (n_weeks, steps_per_week)
    daa_val_weeks   = load_split("data_splits/daa/manifest.json", "val")
    daa_test_weeks  = load_split("data_splits/daa/manifest.json", "test")
    day_ahead_prices_hourly = daa_train_weeks[0]
    day_ahead_prices_hourly_eval = daa_val_weeks[0]

    # day_ahead_prices_hourly_eval = day_ahead_prices_hourly.iloc[:, 0] + np.random.normal(0, 0.01, size=day_ahead_prices_hourly.shape[0])  # Slightly different profile for evaluation

    day_ahead_market = DaaMarket(price_profile_per_simulation_time_step=pd.Series(day_ahead_prices_hourly, index=timestamps_daa))
    daa_price_forecaster = DataProfileForecaster(forecast_data=day_ahead_prices_hourly,
                                         time_delta_seconds=3600)
    
    day_ahead_market_eval = DaaMarket(price_profile_per_simulation_time_step=pd.Series(day_ahead_prices_hourly_eval, index=timestamps_daa))
    daa_price_forecaster_eval = DataProfileForecaster(forecast_data=day_ahead_prices_hourly_eval,
                                         time_delta_seconds=3600)
    
    # Configure markets
    print("\n2. Configuring markets...")
    
    # Choose one of these configurations:
    
    idc_market = IdcMarket(price_profile_per_simulation_time_step=pd.Series(idc_price_profile, index=timestamps))
    idc_price_forcaster = DataProfileForecaster(idc_price_profile, time_delta_seconds=900)
    
    idc_market_eval = IdcMarket(price_profile_per_simulation_time_step=pd.Series(idc_price_profile_eval, index=timestamps))
    idc_price_forcaster_eval = DataProfileForecaster(idc_price_profile_eval, time_delta_seconds=900)    

    # norm_idc_price = 2 * (idc_price_profile - np.min(idc_price_profile)) / (np.max(idc_price_profile) - np.min(idc_price_profile) + 1e-8) - 1
    # norm_idc_eval_price = 2 * (idc_price_profile_eval - np.min(idc_price_profile_eval)) / (np.max(idc_price_profile_eval) - np.min(idc_price_profile_eval) + 1e-8) - 1

    # idc_market = IdcMarket(price_profile_per_simulation_time_step=pd.Series(norm_idc_price, index=timestamps))    
    # idc_price_forcaster = DataProfileForecaster(norm_idc_price, time_delta_seconds=900)
    
    # idc_market_eval = IdcMarket(price_profile_per_simulation_time_step=pd.Series(norm_idc_eval_price, index=timestamps))
    # idc_price_forcaster_eval = DataProfileForecaster(norm_idc_eval_price, time_delta_seconds=900)
    
    
    markets = MarketsWrapper(
        battery_capacity_kwh=STORAGE_CAPACITY,
        battery_max_power_kwh=STORAGE_POWER,
        intraday_market=idc_market,
        idc_price_forcaster=idc_price_forcaster,  # Placeholder, can be set to actual forecaster instance,
        day_ahead_market=day_ahead_market,
        daa_price_forecaster=daa_price_forecaster,
        max_steps=MAX_EPISODE_STEPS
    )

    markets_eval = MarketsWrapper(
        battery_capacity_kwh=STORAGE_CAPACITY,
        battery_max_power_kwh=STORAGE_POWER,
        intraday_market=idc_market_eval,
        idc_price_forcaster=idc_price_forcaster_eval,  # Placeholder, can be set to actual forecaster instance,
        day_ahead_market=day_ahead_market_eval,
        daa_price_forecaster=daa_price_forecaster_eval,
        max_steps=MAX_EPISODE_STEPS
    )

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
    
    energy_system_eval = EnergySystem(time_index=timestamps)
    grid_eval = Grid(label='grid_eval')
    battery_eval = StorageSystemEnergyReservoir(label='battery',
                                                time_delta_seconds=900,
                                                nom_power=STORAGE_POWER,
                                                capacity=STORAGE_CAPACITY,
                                                initial_soc=0.5,
                                                eta_charge=1,
                                                eta_discharge=1,
                                                )
    energy_system_eval.add_components(grid_eval, battery_eval)

    eval_env = create_env(
        energy_system=energy_system_eval,
        markets=markets_eval,
        idc_price_data=idc_val_weeks,
        daa_price_data=daa_val_weeks,
        forecast_horizon_hours=3,
        time_delta_seconds = TIME_DELTA_SECONDS,
        battery_power_kwh=STORAGE_POWER,
        battery_capacity_kwh=STORAGE_CAPACITY,
        max_episode_steps=MAX_EPISODE_STEPS,  # 24 hours with 15 min steps
        start_date=START_DATE,
    )

    # Create environment
    print("\n3. Creating environment...")
    env = create_env(
        energy_system=energy_system,
        markets=markets,
        idc_price_data=idc_train_weeks,
        daa_price_data=daa_train_weeks,
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

    # model = continue_training(
    #     model_path="models/idc_only/best/best_model.zip",
    #     env=env,
    #     eval_env=eval_env,
    #     total_timesteps=2_000_000,
    #     save_dir="models/continued",
    # )
        
    
    # model = train_ppo_agent(
    #     env=env,
    #     eval_env=eval_env,
    #     total_timesteps=10_000_000,  # Increase for better results
    #     save_path='./models/idc_only',
    #     model_name='ppo_trading',
    #     eval_freq=50_000,
    #     checkpoint_freq=100_000,
    #     use_vec_normalize=False,
    #     verbose=0
    # )

    model = train_sac_agent(
        env=env,
        eval_env=eval_env,
        total_timesteps=1_500_000,  # Increase for better results
        save_path='./models/idc_daa',
        model_name='sac_trading',
        eval_freq=30_000,
        checkpoint_freq=100_000,
        use_vec_normalize=False,
        verbose=0
    )

    # model = train_sac_agent(
    #     energy_system=energy_system,
    #     train_idc=idc_train_weeks,
    #     train_daa=daa_train_weeks,
    #     val_idc=idc_val_weeks,
    #     val_daa=daa_val_weeks,
    #     test_idc=idc_test_weeks,
    #     test_daa=daa_test_weeks,
    #     n_epochs=10,  # Increase for better results
    #     save_path='./models/idc_daa',
    #     model_name='sac_trading',
    #     eval_freq=20_000,
    #     checkpoint_freq=40_000,
    #     use_vec_normalize=True,
    #     verbose=0
    # )
    
    # Evaluate agent
    print("\n6. Evaluating trained agent...")
    results = evaluate_agent(
        model=model,
        env=env,
        n_episodes=10,
        render=False
    )

    # results = load_and_evaluate(
    #     model_path='./models/idc_daa/best/best_model',
    #     test_idc=idc_test_weeks,
    #     test_daa=daa_test_weeks,
    #     energy_system=energy_system,
    #     use_vec_normalize=False,
    # )

    # Plot results
    print("\n7. Plotting results...")
    plot_eval_results(results, save_path='./models/idc_daa/evaluation_results.png')
    
    print("\n" + "=" * 60)
    print("Training pipeline completed!")
    print("=" * 60)



def plot_eval_results(
    results: dict,
    save_path: str = None,
    show: bool = True,
) -> None:
    """
    Plot evaluation results returned by evaluate_policy.
 
    Produces a 2-panel figure:
      Top    — reward per week (bar chart) with mean ± std band
      Bottom — cumulative reward across weeks
 
    Parameters
    ----------
    results   : dict returned by evaluate_policy
    save_path : if provided, saves the figure to this path (e.g. 'eval.png')
    show      : call plt.show() if True
    """
    rewards   = results['rewards']
    mean_r    = results['mean_reward']
    std_r     = results['std_reward']
    n_weeks   = len(rewards)
    weeks     = np.arange(1, n_weeks + 1)
    cumulative = np.cumsum(rewards)
 
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('SAC Agent — Test Set Evaluation', fontsize=14, fontweight='bold')
 
    # ── Top: per-week reward ────────────────────────────────────────────────
    ax = axes[0]
    colors = ['steelblue' if r >= 0 else 'tomato' for r in rewards]
    ax.bar(weeks, rewards, color=colors, alpha=0.8, zorder=2)
    ax.axhline(mean_r, color='black',  linewidth=1.5, linestyle='--', label=f'Mean ({mean_r:.2f})')
    ax.axhspan(mean_r - std_r, mean_r + std_r, alpha=0.12, color='black', label=f'±1 Std ({std_r:.2f})')
    ax.axhline(0, color='grey', linewidth=0.8, linestyle='-')
    ax.set_xlabel('Week')
    ax.set_ylabel('Total Reward')
    ax.set_title('Reward per Test Week')
    ax.set_xticks(weeks)
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3, zorder=1)
 
    # Annotate best/worst weeks
    best_idx  = int(np.argmax(rewards))
    worst_idx = int(np.argmin(rewards))
    ax.annotate(f'Best\n{rewards[best_idx]:.1f}',
                xy=(weeks[best_idx], rewards[best_idx]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=8, color='steelblue')
    ax.annotate(f'Worst\n{rewards[worst_idx]:.1f}',
                xy=(weeks[worst_idx], rewards[worst_idx]),
                xytext=(0, -18), textcoords='offset points',
                ha='center', fontsize=8, color='tomato')
 
    # ── Bottom: cumulative reward ───────────────────────────────────────────
    ax2 = axes[1]
    ax2.plot(weeks, cumulative, color='steelblue', linewidth=2, marker='o',
             markersize=4, zorder=2)
    ax2.fill_between(weeks, cumulative, alpha=0.15, color='steelblue')
    ax2.axhline(0, color='grey', linewidth=0.8)
    ax2.set_xlabel('Week')
    ax2.set_ylabel('Cumulative Reward')
    ax2.set_title(f'Cumulative Reward  (total: {cumulative[-1]:.2f})')
    ax2.set_xticks(weeks)
    ax2.grid(axis='y', alpha=0.3, zorder=1)
 
    plt.tight_layout()
 
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to '{save_path}'")
 
    if show:
        plt.show()
 
    plt.close()


if __name__ == '__main__':
    main()



