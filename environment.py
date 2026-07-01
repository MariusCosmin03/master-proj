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
        idc_price_data: list[pd.Series],
        daa_price_data: list[pd.Series],
        forecast_horizon_hours: int = 3,
        time_delta_seconds: int = 900,
        battery_power_kwh: float = 10.0,
        battery_capacity_kwh: float = 1000.0,
        max_episode_steps: int = 31 * 96,  # 24 hours with 15 min steps
        start_date: pd.Timestamp = pd.Timestamp("2024-01-01 00:00:00"),
        seed: Optional[int] = None,
        ):
        super(TradingEnvironment, self).__init__()
        # self.data = data
        # self.forecaster = forecaster
        self.seed = seed
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

        self.idc_prices = idc_price_data
        self.daa_prices = daa_price_data

        # Define action and observation space
        # ----- Action Space -----
        # [idc_volume_kw] over the next 3 hours (12 time steps of 15 min each)
        self.action_space = spaces.Box(
            low = np.array([-1] * 2, dtype=np.float64),
            high = np.array([1] * 2, dtype=np.float64),
            dtype = np.float64
        )

        # ----- Observation Space -----
        # Energy system state(2): soc(1) 

        # Market state:
        # - Base features (5): time step(normalized), time of day(sin+cos), day of week(sin+cos)
        # - IDC-specific features (3 + 0): current price(1), 1h momentum(1), 3h momentum(1), normalized position(12)
        # - Forecasts(12 + 0): 12 future price forecasts (12) # + confidence levels (24)
        # - Alternate Reward Function(0): # Pending charge volume and cost (2)
        # - Forecasted prices for next day (24): Hourly prices for the next day from DAA (24)
        # - Current DAA promise(1): The current promised schedule from DAA from the previous day (1)

        obs_dim = 1 + (1+ 2 * 2) + (3 + 0) + (4 * 24 + 0) + (24) + (1)  # + 2
        self.markets.idc_horizon_length = 4 * 24  # 24 hours ahead with 15 min intervals
        self.observation_space = spaces.Box(
            low = -np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float64
        )


    def reset(self, *, seed: Optional[int] = None):
        super().reset(seed=self.seed)
        self.current_step = 0

        self.reset_prices_discrete()

        energy_state = self.energy_system.reset()
        market_state, market_info = self.markets.reset()
        
        market_info["soc"] = energy_state["components_states"]["battery"]["soc"]
        market_info["fixed_power_balance"] = energy_state["fixed_power_balance"]
        market_info["initial_soc"] = energy_state["components_states"]["battery"]["soc"]


        assert 0.4 <= energy_state["components_states"]["battery"]["soc"] <= 0.6, \
        f"Battery SoC did not reset: {energy_state['components_states']['battery']['soc']}"

        # print(energy_state, market_state)
        obs = self._build_observation(energy_state, market_state)
        # obs = np.zeros(self.observation_space.shape, dtype=np.float64)  # Placeholder for initial observation
        info = {"energy_state": energy_state, "market_state": market_state}

        return obs, market_info

    def step(self, action:np.ndarray):

        # 1. Translate current market trades into physical action
        
        action = action * self.battery_power_kwh / (4 * 2)  # Scale action to actual kWh for 15 min step, and limit to battery power
                                                            # Divide by 4 for 15 min steps, and by 2 for bidirectional power (charge/discharge)

        # 3. Step the Markets with the market action(trades) -> market state + reward
        market_state, market_reward, market_done, market_info = self.markets.step(action)

        physical_change = self._translate_to_physical_change()

        # 2. Step the EnergySystem with the physical set-point -> energy state + reward
        power_contribution_per_component, energy_state, energy_system_done = self.energy_system.simulate_one_time_step({"battery":physical_change})
        energy_reward = 0

        actual_change = power_contribution_per_component["battery"]
        # print(f"{self.current_step}: Actual change: {actual_change}, Physical change: {physical_change}, Action: {action}, SoC: {energy_state['components_states']['battery']['soc']}")
        execution_gap = physical_change - actual_change     # kWh
        # max_power = self.battery_power_kwh

        # Penalty scales with how badly the battery failed to execute
        # Max penalty ≈ 2× avg step revenue, not 240×
        soc_violation = 0
        curr_price_norm_idc = self.markets.get_idc_price_normalized()

        # market_reward = curr_price_norm_idc * actual_change  # Still reward based on what was actually executed
        idc_correction = execution_gap * curr_price_norm_idc
        market_reward = market_reward - idc_correction  # Penalize the market reward based on the execution gap
        # print(f"Market reward adjusted for execution gap: {market_rew_old:.3f} -> {market_reward:.3f} (gap: {execution_gap:.3f}, price_norm: {curr_price_norm_idc:.3f})")
        if execution_gap > 0.0001:  # Allow small execution errors
            # energy_reward = -10 # make smaller
            soc_violation = 1
        else:
            energy_reward = 0.0

        if energy_state is not None:
            soc = energy_state["components_states"]["battery"]["soc"]

        # Soft quadratic SoC penalty — activates near limits, zero in safe zone
        # soc_penalty = 0.0
        # if soc > 0.85:
        #     soc_penalty = -((soc - 0.85) / 0.15) ** 2 * 50    # up to -€50 at SoC=1.0
        # elif soc < 0.15:
        #     soc_penalty = -((0.15 - soc) / 0.15) ** 2 * 50

        # Alignment bonus: reward holding low SoC when price is high (space to discharge)
        # and high SoC when price is low (energy ready to sell later)
        # This teaches the agent to *preserve optionality*
        # alignment_bonus = -price_norm * (soc - 0.5) * 10

        # energy_reward += soc_penalty # + alignment_bonus

        # 4. Combine state and reward
            combined_state = self._build_observation(energy_state, market_state)
        else:
            print("Warning: Energy state is None. Using market state only for observation and zero reward. Time step:", self.current_step, "Energy done was:", energy_system_done)
            soc = 0  # Default SoC if energy state is unavailable
            combined_state = np.concatenate([np.array([soc]), market_state])  # Placeholder if energy state is unavailable

        reward = float(energy_reward + market_reward)

        self.current_step += 1
        done = self.current_step >= self.max_episode_steps -1  or market_done or energy_system_done
        idc_done_trade_sellout = 0
        if done:
            # print(f"Episode done at step {self.current_step}. Market done: {market_done}, Energy system done: {energy_system_done}")
            reward = reward + soc * self.battery_capacity_kwh * curr_price_norm_idc   # Final reward bonus for remaining SoC and power balance at episode end
            idc_done_trade_sellout = soc * self.battery_capacity_kwh # amount of energy sold at the end of the episode for the idc price at the last timepoint
            # print(f"Final reward bonus for remaining SoC: {soc * self.battery_capacity_kwh * price:.3f} (SoC: {soc:.3f}, Price: {price:.3f})")

        

        # Sanity checks for NaN values in the observation and reward
        if np.any(np.isnan(combined_state)):
            print(f"WARNING: NaN in observation: {combined_state}")
        if np.isnan(reward):
            print(f"WARNING: NaN reward at step")


        # print(combined_state)
        
        market_info["energy_reward"] = energy_reward
        market_info["market_reward"] = market_reward

        market_info["soc"]              = soc # [0,1]
        market_info["execution_gap"]     = execution_gap
        market_info["idc_trade"] = market_info.get("idc_trade_initial", 0.0) - execution_gap + idc_done_trade_sellout
        market_info["idc_correction"] = idc_correction
        market_info["constraint_fired"] = int(soc_violation)

        # print(f"Action: {action[0]:.3f} | Physical: {physical_change:.3f} | Actual: {actual_change:.3f} | SoC: {soc:.3f}")

        market_info["final_reward"] = reward
        market_info["idc_done_trade_sellout"] = idc_done_trade_sellout

        return combined_state, reward, done, False, market_info # state, reward, done, truncated, info
    
    def _translate_to_physical_change(self):
        """
        Convert market trades into a battery charge/discharge for CURRENT TIME STEP [kWh] 

        The net traded volume determines wheter the battery needs to 
        charge(buying energy) or discharge(selling).
        """
        # start_time = self.start_date + timedelta(minutes=15 * self.current_step)
        
        power_contribution = self.markets.get_total_current_contribution()
        # print(f"Physical change: {power_contribution:.3f} kWh at step {self.current_step}")
        return power_contribution
    
    def _build_observation(self, energy_state, market_state):
        """
        Flatten and concatenate energy and market states into a single vector.
        """
        soc = energy_state["components_states"]["battery"]["soc"]
        # fixed_power_balance = energy_state["fixed_power_balance"]
        # fixed_power_balance_norm = fixed_power_balance / self.battery_capacity_kwh

        
        soc_normalized = (soc - 0.5) * 2  # 0.1 → -0.8, 0.9 → 0.8
        
        energy_obs = np.array(
            [
                soc_normalized, # fixed_power_balance_norm,
            #  energy_state["fixed_power_contribution_per_component"]["battery"],
             ], dtype=np.float64
        )
     
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
    
    def reset_prices(self): #, new_starting_date: pd.Timestamp):
        """
        Reset the market prices to new values based on a random starting date.
        """
        new_starting_date = np.random.randint(0, len(self.idc_prices) - self.max_episode_steps)
        new_idc_prices = self.idc_prices[new_starting_date:new_starting_date + self.max_episode_steps]
        new_daa_prices = self.daa_prices[new_starting_date:new_starting_date + self.max_episode_steps//4]  # Assuming DAA prices are hourly
        self.markets.reset_prices(new_idc_prices, new_daa_prices)
        self.energy_system.time_index = new_idc_prices.index  # Update the energy system's time index to match the new prices

    def reset_prices_discrete(self):
        """
        Reset the market prices to new values based on a random starting date, ensuring that the starting date aligns with discrete time steps.
        """
        idx = np.random.randint(0, len(self.idc_prices))

        new_idc_prices = self.idc_prices[idx]
        new_daa_prices = self.daa_prices[idx]
        self.markets.reset_prices(new_idc_prices, new_daa_prices)
        self.energy_system._time_index = new_idc_prices.index  # Update the energy system's time index to match the new prices


