from __future__ import annotations
from datetime import timedelta

import numpy as np
from pyparsing import Optional

# Prefer gymnasium if available (SB3 supports it), fallback to gym
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYMNASIUM = True
except ImportError:
    import gym
    from gym import spaces
    _GYMNASIUM = False

from nrgise import EnergySystem
import pandas as pd



class BaseEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning."""

    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        # data: np.ndarray,
        # forecaster: pd.Series,
        energy_system: EnergySystem,
        forecast_horizon_hours: int = 3,
        time_delta_seconds: int = 900,
        battery_power_kwh: float = 10.0,
        battery_capacity_kwh: float = 1000.0,
        max_episode_steps: int = 7 * 96,  # 24 hours with 15 min steps
        start_date: pd.Timestamp = pd.Timestamp("2024-01-01 00:00:00"),
        ):
        super(BaseEnvironment, self).__init__()
        # self.data = data
        # self.forecaster = forecaster
        self.energy_system = energy_system
        self.current_step = 0
        self.forecast_horizon_hours = forecast_horizon_hours
        self.time_delta_seconds = time_delta_seconds
        self.forecast_horizon = forecast_horizon_hours * 3600 // time_delta_seconds
        self.battery_power_kwh = battery_power_kwh
        self.battery_capacity_kwh = battery_capacity_kwh

        self.max_episode_steps = max_episode_steps

        self.start_date = start_date

        # Define action and observation space
        # ----- Action Space -----
        # [idc_volume_kw] over the next 3 hours (12 time steps of 15 min each)
        self.action_space = spaces.Box(
            low = np.array([-battery_power_kwh / 2] * 1, dtype=np.float64),
            high = np.array([battery_power_kwh / 2] * 1, dtype=np.float64),
            dtype = np.float64
        )

        # ----- Observation Space -----
        # Energy system state: 
        #  - soc(1) + pv_output_kw(1) + 1
    

        # State generic:
        # - Temporal features (5): time step, time of day, day of week
        obs_dim = 7 #(pending_charge_volume, pending_charge_cost)
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )


    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self._current_step = 0

        energy_state = self.energy_system.reset()
        
        info = {}
        info["soc"] = float(energy_state["components_states"]["battery"]["soc"])
        info["fixed_power_balance"] = energy_state["fixed_power_balance"]
        info["initial_soc"] = energy_state["components_states"]["battery"]["soc"]

        assert 0.4 <= energy_state["components_states"]["battery"]["soc"] <= 0.6, \
        f"Battery SoC did not reset: {energy_state['components_states']['battery']['soc']}"

        
        obs = self._build_observation(energy_state)
    
        return obs, info

    def step(self, action:np.ndarray):


        # 2. Step the EnergySystem with the physical set-point -> energy state + reward
        power_contribution_per_component, energy_state, energy_system_done = self.energy_system.simulate_one_time_step({"battery":action})
        
        info = {}
        if not energy_system_done:
            info["soc"] = float(energy_state["components_states"]["battery"]["soc"])
            info["fixed_power_balance"] = energy_state["fixed_power_balance"]
            info["norm_step"] = self._current_step / self.max_episode_steps
        else:
            info["soc"] = -1.0  # Indicate terminal state with invalid SoC
            info["fixed_power_balance"] = 0
            info["norm_step"] = 0

        reward = self._calculate_reward(energy_state)
        
        next_obs = self._build_observation(energy_state, )
        self._current_step += 1
        terminated = energy_system_done
        truncated = self._current_step >= self.max_episode_steps
        

        return next_obs, reward, terminated, truncated, info # state, reward, done, truncated, info
    

    def _calculate_reward(self, energy_state) -> float:
        """
        Calculate reward based on energy state.
        """
        # Placeholder for reward calculation logic
        # Example: reward = -info["current_price"] * action[0]  # Cost of energy traded
        reward = float(np.random.rand()) - 0.5
        return reward * 2

    
    
    def _build_observation(self, energy_state) -> np.ndarray:
        """
        Flatten and concatenate energy states into a single vector.
        """

        if energy_state is None:
            return np.zeros(self.observation_space.shape, dtype=np.float64)  # Placeholder for observation when energy state is not available
        
        # step_idx = min(self._current_step, len(self.energy_system._time_index) - 1)
        timestamp : pd.Timestamp = pd.Timestamp(self.energy_system._time_index[self._current_step])
        norm_step = self._current_step / self.max_episode_steps
        
        # Add time features (e.g., time of day, day of week)
        hour = timestamp.hour + timestamp.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        # Day of week (P = 7, where Monday = 0):
        day_of_week = timestamp.day_of_week
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        energy_obs = np.array(
            [float(energy_state["components_states"]["battery"]["soc"]), 
             float(energy_state["fixed_power_balance"]),
             # energy_state["fixed_power_contribution_per_component"]["battery"],
             
             ], dtype=np.float64
        )
        energy_obs = energy_obs.flatten()
        energy_obs = np.concatenate([energy_obs, [norm_step, hour_sin, hour_cos, day_sin, day_cos]])
     
        return energy_obs


    def render(self, mode='human'):
        pass

    def add_component(self, component):
        # Placeholder for adding a component to the environment
        self.energy_system.add_component(component)

    def get_component_by_label(self, label):
        # Placeholder for retrieving a component by its label
        return self.energy_system.get_component_by_label(label)
    
    def get_grid_builder(self):
        # Placeholder for retrieving the grid builder
        return self.energy_system.get_grid_builder()
    
    def get_system_components(self):
        # Placeholder for retrieving all components in the system
        return self.energy_system.components()
    
    def get_controlable_components(self):
        # Placeholder for retrieving controllable components
        return self.energy_system.controllable_components()
    
    def get_time_delta_seconds(self):
        # Placeholder for retrieving the time delta in seconds
        return self.energy_system.time_delta_seconds()