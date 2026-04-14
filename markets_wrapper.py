from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from markets.idc_market import IdcMarket, IdcTradeSequence
from nrgise import Forecaster


@dataclass
class Forcasters:
    idc: Optional[Forecaster] = None

class MarketsWrapper:


    def __init__(
            self, 
            initial_balance: float,
            battery_capacity_kwh: float,
            battery_max_power_kwh: float,
            # day_ahead_market, 
            intraday_market: IdcMarket,
            idc_price_forcaster,
            max_steps: int = 96 * 30,  # 7 days with 15 min steps
    ):
        self.idc = intraday_market
        self.initial_balance = initial_balance
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_power_kwh = battery_max_power_kwh
        # self.daa = day_ahead_market

        self._pending_charge_cost = 0.0
        self._pending_charge_volume = 0.0

        self.forecasters = Forcasters(idc=idc_price_forcaster,
                                      # daa = daa_price_forecaster
                                      )
        
        self.time_step = 0
        self.max_steps = max_steps
    
    
    def place_idc_trade_seq(self, trade_sequence: IdcTradeSequence):
        # Placeholder for placing a sequence of trades in the market
        self.idc.place_trade_sequence(trade_sequence)


    def step(self, action):
        # Placeholder for executing a step in the market based on the action taken by the agent
        # This would involve placing trades, updating market state, and calculating rewards
        

        length_of_action = self.idc.get_valid_trade_sequence_length_for_now()
        idc_action = np.pad(action, (0, length_of_action - len(action)), mode="constant")
        # print(f"Placing IDC trade sequence: {idc_action}, valid length: {length_of_action}, action shape: {idc_action.shape}")
        self.idc.place_trade_sequence(IdcTradeSequence(idc_action))
        
        
        
        self.idc.handle_time_step_update(self.time_step + 1)

        reward = self.reward(self.idc.time_step) # self.idc.get_revenue() / self.initial_balance

        self.time_step += 1
        

        return self.get_state(), reward, False, {"idc_reward": reward}  # state, reward, done, info

    def reset(self):
        # Placeholder for resetting the market to an initial state at the beginning of an episode
        self.idc.reset()
        self.time_step = 0
        self._pending_charge_cost = 0.0
        self._pending_charge_volume = 0.0
        state = self.get_state()
        return np.insert(state, 8, np.zeros(12)), {}  # initial state, info 
                                # include 0 in place for the normalized position which will be filled in the first step

    def get_state(self):
        """
        Generate a state representation for the markets, including current prices, forecasts, confiddence levels,
        and any other relevant information.
        """

        # Normalize features
        # norm_balance = self.idc.get_balance() / self.initial_balance
        timestamp : pd.Timestamp = self.idc.data_profile.index[self.time_step]
        norm_step = self.idc.time_step / self.max_steps

        state_info = np.array([norm_step], dtype=np.float32)

        # Add time features (e.g., time of day, day of week)
        

        hour = timestamp.hour + timestamp.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        # Day of week (P = 7, where Monday = 0):
        day_of_week = timestamp.day_of_week # Sunday
        day_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_cos = np.cos(2 * np.pi * day_of_week / 7)

        state_info = np.append(state_info, 
                               [hour_sin, hour_cos, 
                                day_sin, day_cos,
                                self._pending_charge_volume / (self.battery_capacity_kwh + 1e-8), 
                                self._pending_charge_cost])

        if self.idc is not None:
            idc_state = self._get_idc_market_state()
            state_info = np.append(state_info, idc_state)

        return state_info
    
    def reward(self, time_step) -> float:
        timestamp = self.idc.data_profile.index[time_step]
        price = self.idc.prices_log[timestamp]
        trade = self.idc.realized_trades_log[timestamp]

        if trade < 0:   # charging — defer cost, no immediate reward
            self._pending_charge_cost += price * abs(trade)
            self._pending_charge_volume += abs(trade)
            base_reward = 0.0
        elif trade > 0:  # discharging — realise full cycle profit
            avg_charge_price = (self._pending_charge_cost / self._pending_charge_volume
                                if self._pending_charge_volume > 0 else 0.0)
            cycle_spread = price - avg_charge_price
            base_reward = cycle_spread * trade
            # Reset buffer proportionally
            self._pending_charge_cost *= max(0, 1 - trade / (self._pending_charge_volume + 1e-8))
            self._pending_charge_volume = max(0, self._pending_charge_volume - trade)
        else:
            base_reward = 0.0

        # Still add SoC penalty from Option 2 to prevent pinning
        return base_reward 
        return price * trade
    
    def get_total_current_contribution(self):
        # Placeholder for calculating the total current contribution of the agent across all markets
        schedule = self.idc.get_cleared_schedule()
        if schedule is not None and not schedule.empty:
            return schedule.iloc[0]
        return 0.0

    def _get_idc_market_state(self):
        state_info = [self.idc.get_price_of_current_time_step() / 85.0]  # Normalize by a reasonable max price

        
        # Price momentum (last 4 intervals) - 1 hour
        if self.idc.time_step >= 4:
            recent_prices = self.idc.data_profile[self.idc.time_step - 4:self.idc.time_step]
            price_change_1h = (self.idc.get_price_of_current_time_step() - recent_prices[0]) / (recent_prices[0] +1e-8)
        else: 
            price_change_1h = 0.0
        state_info = np.append(state_info, price_change_1h)

        # Price momentum (last 12 intervals) - 3 hours
        if self.idc.time_step >= 12:
            recent_prices = self.idc.data_profile[self.idc.time_step - 12:self.idc.time_step]
            price_change_3h = (self.idc.get_price_of_current_time_step() - recent_prices[0]) / (recent_prices[0] +1e-8)
        else: 
            price_change_3h = 0.0

        state_info = np.append(state_info, price_change_3h)

        # Market-specific features
        norm_position = self.idc.get_cleared_schedule().iloc[:12] * (4 / self.battery_max_power_kwh)
        
        
        # norm_pending = self.idc.get_revenues_per_time_step() / self.initial_balance
            
        # state_info = np.append(state_info, norm_pending)
        
        # Append normalized forecasts
        idc_forecast, idc_confidence = self.get_idc_forecast()
        norm_forecast = idc_forecast / 85.0
        state = np.concatenate([state_info, norm_position, norm_forecast, idc_confidence])
        
        return np.array(state, dtype=np.float32)
    
    def get_idc_forecast(self):
        horizon_length = 12 # 3 hours ahead with 15 min intervals
        return self.forecasters.idc.predict(horizon_length), np.ones(horizon_length) # Placeholder confidence levels

    def get_idc_trades(self):
        # Placeholder for retrieving current trades in the intraday market
        return self.idc.get_realized_trades()
    
