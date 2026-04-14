from __future__ import annotations
from datetime import timedelta

import numpy as np
from pyparsing import Optional

from markets_wrapper import MarketsWrapper

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



class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning."""

    metadata = {'render.modes': ['human']}

    def __init__(
        self, 
        # data: np.ndarray,
        # forecaster: pd.Series,
        energy_system: EnergySystem,
        markets: MarketsWrapper,
        forecast_horizon_hours: int = 3,
        time_delta_seconds: int = 900,
        battery_power_kwh: float = 10.0,
        battery_capacity_kwh: float = 1000.0,
        max_episode_steps: int = 31 * 96,  # 24 hours with 15 min steps
        start_date: pd.Timestamp = pd.Timestamp("2024-01-01 00:00:00"),
        ):
        super(TradingEnvironment, self).__init__()
        # self.data = data
        # self.forecaster = forecaster
        self.energy_system = energy_system
        self.markets = markets
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
            low = np.array([-battery_power_kwh / 4] * 1, dtype=np.float64),
            high = np.array([battery_power_kwh / 4] * 1, dtype=np.float64),
            dtype = np.float64
        )

        # ----- Observation Space -----
        # Energy system state: 
        #  - soc(1) + pv_output_kw(1) + 1
    

        # Market state:
        # - Base features (3): time step, time of day, day of week
        # - IDC-specific features (16): current price(1), 1h momentum(1), 3h momentum(1), revenue(1), # normalized position(12)
        # - Forecasts: 12 future price forecasts + confidence levels (24)
        obs_dim = 1+ 2 * 2 + 3 + 14 + 24 +2 #(pending_charge_volume, pending_charge_cost)
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )


    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=seed)
        self._current_step = 0

        energy_state = self.energy_system.reset()
        market_state, market_info = self.markets.reset()
        
        market_info["soc"] = energy_state["components_states"]["battery"]["soc"]
        market_info["fixed_power_balance"] = energy_state["fixed_power_balance"]

        assert 0.4 <= energy_state["components_states"]["battery"]["soc"] <= 0.6, \
        f"Battery SoC did not reset: {energy_state['components_states']['battery']['soc']}"

        print(energy_state, market_state)
        obs = self._build_observation(energy_state, market_state)
        # obs = np.zeros(self.observation_space.shape, dtype=np.float64)  # Placeholder for initial observation
        info = {"energy_state": energy_state, "market_state": market_state}
    
        return obs, info

    def step(self, action:np.ndarray):
        # 1. Translate current market trades into physical action
        physical_change = self._translate_to_physical_change()
        

        # 3. Step the Markets with the market action(trades) -> market state + reward
        market_state, market_reward, market_done, market_info = self.markets.step(action)

        # 2. Step the EnergySystem with the physical set-point -> energy state + reward
        power_contribution_per_component, energy_state, energy_system_done = self.energy_system.simulate_one_time_step({"battery":physical_change})
        # energy_reward = 0
        # if power_contribution_per_component["battery"] != physical_change:
        #     energy_reward = -10000
        actual_change = power_contribution_per_component["battery"]
        execution_gap = abs(actual_change - physical_change)        # kWh
        max_power = self.battery_power_kwh

        # Penalty scales with how badly the battery failed to execute
        # Max penalty ≈ 2× avg step revenue, not 240×
        soc_violation = 0
        curr_price = self.markets.idc.get_price_of_current_time_step()
        if execution_gap > 0.0001:  # Allow small execution errors
            energy_reward = -1000 # make smaller
            market_reward = curr_price * actual_change  # Still reward based on what was actually executed
            soc_violation = 1
        else:
            energy_reward = 0.0

        soc = energy_state["components_states"]["battery"]["soc"]
        price = self.markets.idc.get_price_of_current_time_step()
        price_norm = (price - 85) / 15   # rough normalisation around mean, adjust to your data

        # Soft quadratic SoC penalty — activates near limits, zero in safe zone
        soc_penalty = 0.0
        if soc > 0.85:
            soc_penalty = -((soc - 0.85) / 0.15) ** 2 * 50    # up to -€50 at SoC=1.0
        elif soc < 0.15:
            soc_penalty = -((0.15 - soc) / 0.15) ** 2 * 50

        # Alignment bonus: reward holding low SoC when price is high (space to discharge)
        # and high SoC when price is low (energy ready to sell later)
        # This teaches the agent to *preserve optionality*
        # alignment_bonus = -price_norm * (soc - 0.5) * 10

        # energy_reward += soc_penalty # + alignment_bonus

        # 4. Combine state and reward
        combined_state = self._build_observation(energy_state, market_state)
        reward = float(energy_reward + market_reward)
        market_info["energy_reward"] = energy_reward

        market_info["soc"]              = energy_state["components_states"]["battery"]["soc"]
        market_info["fixed_power_balance"] = energy_state["fixed_power_balance"]
        market_info["price"]            = self.markets.idc.get_price_of_current_time_step()
        market_info["trade"]            = self.markets.idc.realized_trades_log.get(
                                            self.markets.idc.time_stamp, 0.0)
        market_info["constraint_fired"] = int(soc_violation)

        self._current_step += 1
        done = self._current_step >= self.max_episode_steps -1  or market_done or energy_system_done

        return combined_state, reward, done, False, market_info # state, reward, done, truncated, info
    
    def _translate_to_physical_change(self):
        """
        Convert market trades into a battery charge/discharge for CURRENT TIME STEP [kWh] 

        The net traded volume determines wheter the battery needs to 
        charge(buying energy) or discharge(selling).
        """
        # start_time = self.start_date + timedelta(minutes=15 * self._current_step)
        
        power_contribution = self.markets.get_total_current_contribution()
        
        return power_contribution
    
    def _build_observation(self, energy_state, market_state):
        """
        Flatten and concatenate energy and market states into a single vector.
        """
        energy_obs = np.array(
            [energy_state["components_states"]["battery"]["soc"], 
             energy_state["fixed_power_balance"], 
            #  energy_state["fixed_power_contribution_per_component"]["battery"],
             ], dtype=np.float64
        )
        energy_obs = energy_obs.flatten()
        
     
        return np.concatenate([energy_obs, market_state])

    
    def _get_forecast(self):
        """
        Get imperfect forecasts for the next few time steps, along with confidence levels.
        Returns:
            forecast (np.ndarray): Forecasted values for the next time steps.
            confidence (np.ndarray): Confidence levels for each forecasted value.
        """

        future_steps = min(self.forecast_horizon, 
            len(self.forecaster) - self.current_step - 1)
        
        if future_steps <= 0:
            return np.zeros(self.forecast_horizon), np.zeros(self.forecast_horizon)
        
        # True future values
        true_future = self.forecaster[self.current_step:self.current_step + future_steps]

        # Add forecast noise
        noise = np.random.randn(future_steps) * self.forecast_noise * true_future
        forecast = true_future + noise

        # Confidence decreases with time
        confidence = np.exp(-np.arange(future_steps) * 0.2)

        # Pad if needed
        if future_steps < self.forecast_horizon:
            pad_length = self.forecast_horizon - future_steps
            forecast = np.pad(forecast, (0, pad_length), constant_values=forecast[-1])
            confidence = np.pad(confidence, (0, pad_length), constant_values=0.1)
        
        return forecast, confidence
    
    def _get_state(self):
        # Placeholder for state representation logic
        energy_sysystem_state = self.energy_system.get_state()
        market_state = self.markets.get_state()

        merged_state = np.concatenate([energy_sysystem_state, market_state])
        return merged_state

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