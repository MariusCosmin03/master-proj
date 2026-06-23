from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from markets.idc_market import IdcMarket, IdcTradeSequence
from markets.daa_market import DaaMarket, DaaTradeSequence
from nrgise import Forecaster
from nrgise.forecaster import DataProfileForecaster


@dataclass
class Forcasters:
    idc: Optional[Forecaster] = None
    daa: Optional[Forecaster] = None

class MarketsWrapper:


    def __init__(
            self, 
            battery_capacity_kwh: float,
            battery_max_power_kwh: float,
            day_ahead_market: DaaMarket,
            intraday_market: IdcMarket,
            idc_price_forcaster,
            daa_price_forecaster,
            max_steps: int = 4 * 24 * 7,  # 7 days with 15 min steps
    ):
        self.idc = intraday_market
        # self.initial_balance = initial_balance
        self.battery_capacity_kwh = battery_capacity_kwh
        self.battery_max_power_kwh = battery_max_power_kwh
        
        self.daa = day_ahead_market

        self._pending_charge_cost = 0.0
        self._pending_charge_volume = 0.0

        self.forecasters = Forcasters(idc=idc_price_forcaster,
                                      daa = daa_price_forecaster
                                      )
        

        self.price_min = self.idc.data_profile.min()
        self.price_max = self.idc.data_profile.max()
        self.price_min_daa = self.daa.data_profile.min()
        self.price_max_daa = self.daa.data_profile.max()
        self._pending_daa_schedule = np.array([])  # Buffer to hold pending DAA schedule until it can be executed in IDC
        self._current_daa_promise = np.zeros(24 * 4)  # To track the current promised schedule from DAA for state representation
        self.schedule_storage = pd.Series() # Store schedules for each market and time step
        self.schedule_pv = pd.Series()

        self.time_step = 0
        self.daa_time_step = 0
        self.max_steps = max_steps
    
    
    def place_idc_trade_seq(self, trade_sequence: IdcTradeSequence):
        # Placeholder for placing a sequence of trades in the market
        self.idc.place_trade_sequence(trade_sequence)


    def step(self, action):
        # Placeholder for executing a step in the market based on the action taken by the agent
        # This would involve placing trades, updating market state, and calculating rewards
        
        idc_action = action[:1]
        daa_action = action[1:]
        timestamp : pd.Timestamp = self.idc.data_profile.index[self.time_step]

        ###### IDC ACTION EXECUTION ######
        length_of_action = self.idc.get_valid_trade_sequence_length_for_now()
        idc_action = np.pad(idc_action, (0, length_of_action - len(idc_action)), mode="constant")
        # print(f"Placing IDC trade sequence: {idc_action}, valid length: {length_of_action}, action shape: {idc_action.shape}")
        self.idc.place_trade_sequence(IdcTradeSequence(idc_action))


       

        ##### DAA ACTION EXECUTION ######
        self._pending_daa_schedule = np.append(self._pending_daa_schedule, daa_action)

        if timestamp.hour == 12 and timestamp.minute == 0:  # Assuming DAA market clears at noon for the next day
            if self._pending_daa_schedule.shape[0] < 24:
                # Sanity check
                if self.time_step > 100:
                    print(f"Warning: DAA action length {self._pending_daa_schedule.shape[0]} is less than 24. Padding with zeros.")
                # Pad with zeros if the action does not provide a full 24-hour schedule
                self._pending_daa_schedule = np.pad(self._pending_daa_schedule, (24 - self._pending_daa_schedule.shape[0], 0), mode="constant")
            daa_bids_hourly = self._pending_daa_schedule[:24]  # Take the first 24 values for the next day's schedule
            self.daa.place_trade_sequence_for_next_day(DaaTradeSequence(daa_bids_hourly))
            self._pending_daa_schedule = np.array([])
            self._current_daa_promise = np.repeat(daa_bids_hourly, 4) / 4  # Update current DAA promise for state representation
            


        self.idc.handle_time_step_update(self.time_step + 1)
        if self.time_step % 4 == 0 and self.daa_time_step * 4 < self.max_steps:  # Every hour, update DAA market state
            self.daa.handle_time_step_update(self.daa_time_step)
            self.daa_time_step += 1

        reward = self.reward(self.idc.time_step) # self.idc.get_revenue() / self.initial_balance

        self.time_step += 1
        
        market_info = {"idc_reward": reward} 
        market_info["daa_price"] = self.daa.get_cleared_market_prices(timestamp.date())[timestamp.hour] if self.daa.get_cleared_market_prices(timestamp.date()) is not None else 0.0
        market_info["daa_trade"] = self.daa.get_cleared_trade_sequence(timestamp.date())[timestamp.hour] if self.daa.get_cleared_trade_sequence(timestamp.date()) is not None else 0.0
        return self.get_state(), reward, False, market_info # state, reward, done, info

    def reset(self):
        # Placeholder for resetting the market to an initial state at the beginning of an episode
        self.idc.reset()
        self.daa.reset()
        self.time_step = 0
        self.daa_time_step = 0
        self._pending_charge_cost = 0.0
        self._pending_charge_volume = 0.0

        self._pending_daa_schedule = np.array([])  # Reset DAA schedule buffer
        self._current_daa_promise = np.zeros(24 * 4)  # Reset current DAA promise
        state = self.get_state()

        
        # return np.insert(state, 8, np.zeros(12)), {}  # initial state, info 
                                # include 0 in place for the normalized position which will be filled in the first step
        return state, {}  # initial state, info

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
                                # self._pending_charge_volume / (self.battery_capacity_kwh + 1e-8), 
                                # self._pending_charge_cost
                                ])

        if self.idc is not None:
            idc_state = self._get_idc_market_state()
            state_info = np.append(state_info, idc_state)

        if self.daa is not None:
            daa_state = self._get_daa_market_state()
            state_info = np.append(state_info, daa_state)

        return state_info
    
    def reward(self, time_step) -> float:
        timestamp = self.idc.data_profile.index[time_step]
        price = self.idc.prices_log[timestamp]
        trade = self.idc.realized_trades_log[timestamp]

        curr_price_norm_idc = 2 * (price - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        


        ################# DAA REWARD OPTION 1: BASED ON REALIZED PROFIT/LOSS #################
        # daa_trades = self.daa.get_cleared_trade_sequence(timestamp.date())
        # price_daa = daa_trades[timestamp.hour]
        # trade_daa = self.daa.prices_per_hour_log[timestamp.date()][timestamp.hour]

        price_daa = self.daa.get_cleared_market_prices(timestamp.date())[timestamp.hour] if self.daa.get_cleared_market_prices(timestamp.date()) is not None else 0.0
        trade_daa = self.daa.get_cleared_trade_sequence(timestamp.date())[timestamp.hour] / 4 if self.daa.get_cleared_trade_sequence(timestamp.date()) is not None else 0.0


        curr_price_norm_daa = 2 * (price_daa - self.price_min_daa) / (self.price_max_daa - self.price_min_daa + 1e-8) - 1

        return curr_price_norm_idc * trade + curr_price_norm_daa * trade_daa

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
        # return base_reward 
        

    def get_idc_price_normalized(self):
        curr_price = self.idc.get_price_of_current_time_step()
        curr_price_norm = 2 * (curr_price - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        return curr_price_norm
    
    def get_total_current_contribution(self):
        # Placeholder for calculating the total current contribution of the agent across all markets
        idc_contribution = 0.0
        schedule = self.idc.get_cleared_schedule()
        if schedule is not None and not schedule.empty:
            idc_contribution = schedule.iloc[0]
        if self.daa is not None:
            daa_schedule = self.daa.get_cleared_schedule()
            if daa_schedule is not None and not daa_schedule.empty:
                return idc_contribution + daa_schedule.iloc[0]
        return idc_contribution

    def _get_idc_market_state(self):
        curr_price_norm = self.get_idc_price_normalized()
        state_info = [curr_price_norm]  # Normalize by a reasonable max price

        
        # Price momentum (last 4 intervals) - 1 hour
        if self.idc.time_step >= 4:
            recent_prices = self.idc.data_profile.iloc[self.idc.time_step - 4:self.idc.time_step]
            price_change_1h = (self.idc.get_price_of_current_time_step() - recent_prices.iloc[0]) / (recent_prices.iloc[0] +1e-8)
        else: 
            price_change_1h = 0.0
        state_info = np.append(state_info, price_change_1h)

        # Price momentum (last 12 intervals) - 3 hours
        if self.idc.time_step >= 12:
            recent_prices = self.idc.data_profile.iloc[self.idc.time_step - 12:self.idc.time_step]
            price_change_3h = (self.idc.get_price_of_current_time_step() - recent_prices.iloc[0]) / (recent_prices.iloc[0] +1e-8)
        else: 
            price_change_3h = 0.0

        state_info = np.append(state_info, price_change_3h)

        # Append normalized forecasts
        idc_forecast, idc_confidence = self.get_idc_forecast()
        # idc_forecast = np.insert(idc_forecast, 0, curr_price)  # Include current price for normalization
        # norm_forecast = np.diff(np.log(idc_forecast))
        norm_forecast = 2 * (idc_forecast - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        # norm_forecast = idc_forecast
        state = np.concatenate([state_info, norm_forecast, ])# idc_confidence])
        
        return np.array(state, dtype=np.float32)
    
    def get_idc_forecast(self):
        horizon_length = 12 # 3 hours ahead with 15 min intervals
        return self.forecasters.idc.predict(horizon_length), np.ones(horizon_length) # Placeholder confidence levels

    def get_idc_trades(self):
        # Placeholder for retrieving current trades in the intraday market
        return self.idc.get_realized_trades()
    
    def _get_daa_market_state(self):


        next_day_prices = self.daa._get_hourly_prices_for_tomorrow()
        if next_day_prices is None or len(next_day_prices) == 0 or np.isnan(next_day_prices).any():
            next_day_prices = np.zeros(24)  # Default to zeros if no data available
        current_daa_promise = self._current_daa_promise[0]
        self._current_daa_promise = self._current_daa_promise[1:]  # Shift the promise buffer


        # Placeholder for retrieving the state of the day-ahead market
        return np.array([current_daa_promise] + list(next_day_prices), dtype=np.float32)

    def reset_prices(self, new_idc_prices, new_daa_prices):
        """
        Reset the market prices to new values based on the provided starting date.

        Args:
            new_idc_prices (list): The new IDC prices.
            new_daa_prices (list): The new DAA prices.
        """

        self.idc._price_profile_per_simulation_time_step = new_idc_prices
        self.daa._price_profile_per_simulation_time_step = new_daa_prices
        # self.idc_market = IdcMarket(price_profile_per_simulation_time_step=new_idc_prices)
        # self.idc_price_forcaster = DataProfileForecaster(new_idc_prices, time_delta_seconds=900)


        # self.daa_market = DaaMarket(price_profile_per_simulation_time_step=new_daa_prices)
        # self.daa_price_forcaster = DataProfileForecaster(forecast_data=new_daa_prices,
        #                                                  time_delta_seconds=3600)
        
        # self.forecasters = Forcasters(idc=self.idc_price_forcaster,
        #                               daa = self.daa_price_forcaster
        #                               )
        self.forecasters.idc._forecast_data = np.array(new_idc_prices)
        self.forecasters.daa._forecast_data = np.array(new_daa_prices)

        # Recalculate normalization bounds for the new week — critical for
        # reward() and _get_idc_market_state() to normalize correctly
        self.price_min = self.idc.data_profile.min()
        self.price_max = self.idc.data_profile.max()
        self.price_min_daa = self.daa.data_profile.min()
        self.price_max_daa = self.daa.data_profile.max()

        self.time_step = 0


