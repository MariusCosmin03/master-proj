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
            day_ahead_market: DaaMarket | None,
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
        if self.daa is None:
            self.has_daa = False
        else:
            self.has_daa = True

        self.forecasters = Forcasters(idc=idc_price_forcaster,
                                      daa = daa_price_forecaster
                                      )
        

        self.price_min = self.idc.data_profile.min()
        self.price_max = self.idc.data_profile.max()
        if self.has_daa:
            self.price_min_daa = self.daa.data_profile.min()
            self.price_max_daa = self.daa.data_profile.max()
        self.price_mean = self.idc.data_profile.mean()
        self.price_std = self.idc.data_profile.std()

        self._pending_daa_schedule = np.array([])  # Buffer to hold pending DAA schedule until it can be executed in IDC
        self._current_daa_promise = np.zeros(24 * 4)  # To track the current promised schedule from DAA for state representation
        self.schedule_storage = pd.Series() # Store schedules for each market and time step
        self.schedule_pv = pd.Series()

        self.time_step = 0
        self.daa_time_step = 0
        self._current_daa_trade_vol = 0.0  # To track the current DAA trade for reward calculation
        self.max_steps = max_steps

        self.idc_horizon_length = 4 * 24  # 24 hours ahead with 15 min intervals
    
    
    def place_idc_trade_seq(self, trade_sequence: IdcTradeSequence):
        # Placeholder for placing a sequence of trades in the market
        self.idc.place_trade_sequence(trade_sequence)


    def step(self, action):
        # Placeholder for executing a step in the market based on the action taken by the agent
        # This would involve placing trades, updating market state, and calculating rewards
        
        idc_action = action[:1]
        
        timestamp : pd.Timestamp = self.idc.data_profile.index[self.time_step]

        ###### IDC ACTION EXECUTION ######
        length_of_action = self.idc.get_valid_trade_sequence_length_for_now()
        idc_action = np.pad(idc_action, (0, length_of_action - len(idc_action)), mode="constant")
        # print(f"Placing IDC trade sequence: {idc_action}, valid length: {length_of_action}, action shape: {idc_action.shape}")
        self.idc.place_trade_sequence(IdcTradeSequence(idc_action))


       

        ##### DAA ACTION EXECUTION ######
        if self.has_daa:
            daa_action = action[1:]
            self._pending_daa_schedule = np.append(self._pending_daa_schedule, daa_action)

            if timestamp.hour == 12 and timestamp.minute == 0:  # Assuming DAA market clears at noon for the next day
                if self._pending_daa_schedule.shape[0] < 24*4:
                    # Sanity check
                    if self.time_step > 100:
                        print(f"Warning: DAA action length {self._pending_daa_schedule.shape[0]} is less than 24. Padding with zeros.")
                    # Pad with zeros if the action does not provide a full 24-hour schedule
                    self._pending_daa_schedule = np.pad(self._pending_daa_schedule, (24*4 - self._pending_daa_schedule.shape[0], 0), mode="constant")
                daa_bids_hourly = self._pending_daa_schedule[:24 * 4].reshape(-1, 4).mean(axis=1)  # Take the first 24 values for the next day's schedule
                self.daa.place_trade_sequence_for_next_day(DaaTradeSequence(daa_bids_hourly))
                
                self._current_daa_promise = self._pending_daa_schedule[:24 * 4]  # Update current DAA promise for state representation
                self._pending_daa_schedule = np.array([])


        self.idc.handle_time_step_update(self.time_step + 1)
        if self.has_daa and self.time_step % 4 == 0 and self.daa_time_step * 4 < self.max_steps:  # Every hour, update DAA market state
            self.daa.handle_time_step_update(self.daa_time_step)
            self.daa_time_step += 1

        reward, market_info = self.reward(self.idc.time_step) # self.idc.get_revenue() / self.initial_balance

        self.time_step += 1
        
        # market_info["daa_price"] = self.daa._price_profile_per_simulation_time_step.iloc[self.daa_time_step-1] if self.daa._price_profile_per_simulation_time_step is not None else 0.0
        # market_info["daa_trade"] = self._current_daa_promise[0]
        
        return self.get_state(), reward, False, market_info # state, reward, done, info

    def reset(self):
        # Placeholder for resetting the market to an initial state at the beginning of an episode
        self.idc.reset()
        if self.has_daa:
            self.daa.reset()
        self.time_step = 0
        self.daa_time_step = 0

        self._pending_daa_schedule = np.array([])  # Reset DAA schedule buffer
        self._current_daa_promise = np.zeros(24 * 4)  # Reset current DAA promise
        self._current_daa_trade_vol = 0.0  # Reset current DAA trade volume
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

        if self.has_daa:
            daa_state = self._get_daa_market_state()
            state_info = np.append(state_info, daa_state)

        return state_info
    
    def reward(self, time_step) -> float:
        timestamp = self.idc.data_profile.index[time_step]
        price = self.idc.prices_log[timestamp]
        trade = self.idc.realized_trades_log[timestamp]

        # curr_price_norm_idc = 2 * (price - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        curr_price_norm_idc = self._normalize_price(price, market_type='idc')


        ################# DAA REWARD OPTION 1: BASED ON REALIZED PROFIT/LOSS #################
        # daa_trades = self.daa.get_cleared_trade_sequence(timestamp.date())
        # price_daa = daa_trades[timestamp.hour]
        # trade_daa = self.daa.prices_per_hour_log[timestamp.date()][timestamp.hour]
        daa_reward = 0.0
        price_daa = 0.0
        trade_daa = 0.0
        curr_price_norm_daa = 0.0
        if self.has_daa:
            price_daa = self.daa._price_profile_per_simulation_time_step.iloc[self.daa_time_step-1] if self.daa._price_profile_per_simulation_time_step is not None else 0.0
            trade_daa = self._current_daa_promise[0]
            self._current_daa_trade_vol = self._current_daa_promise[0]


            # curr_price_norm_daa = 2 * (price_daa - self.price_min_daa) / (self.price_max_daa - self.price_min_daa + 1e-8) - 1
            curr_price_norm_daa = self._normalize_price(price_daa, market_type='daa')

            
            daa_reward = curr_price_norm_daa * trade_daa
        idc_reward = curr_price_norm_idc * trade
        initial_reward = idc_reward + daa_reward

        market_info = {
            "idc_price": price,
            "idc_trade_initial": trade,
            "daa_price": price_daa,
            "daa_trade": trade_daa,
            "curr_price_norm_idc": curr_price_norm_idc,
            "curr_price_norm_daa": curr_price_norm_daa,
            "idc_reward": idc_reward,
            "daa_reward": daa_reward,
            "initial_reward": initial_reward
        }

        # print(price, trade, price_daa, trade_daa, curr_price_norm_idc, curr_price_norm_daa)
        
        return initial_reward, market_info
        

    def get_idc_price_normalized(self):
        curr_price = self.idc.get_price_of_current_time_step()
        # curr_price_norm = 2 * (curr_price - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        curr_price_norm = self._normalize_price(curr_price, market_type='idc')
        return curr_price_norm
    
    def get_total_current_contribution(self):
        # Placeholder for calculating the total current contribution of the agent across all markets
        idc_contribution = 0.0
        schedule = self.idc.get_cleared_schedule()
        if schedule is not None and not schedule.empty:
            idc_contribution = schedule.iloc[0]
        if self.has_daa:
            # daa_schedule = self.daa.get_cleared_schedule()  # self._current_daa_promise[0]
            # if daa_schedule is not None and not daa_schedule.empty:
            #     return idc_contribution + daa_schedule.iloc[0]
            return idc_contribution + self._current_daa_trade_vol
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
        idc_forecast, idc_confidence = self.get_idc_forecast(self.idc_horizon_length)
        # idc_forecast = np.insert(idc_forecast, 0, curr_price)  # Include current price for normalization
        # norm_forecast = np.diff(np.log(idc_forecast))
        # norm_forecast = 2 * (idc_forecast - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
        norm_forecast = self._normalize_price(idc_forecast, market_type='idc')
        # norm_forecast = self._log_returns_aligned(idc_forecast)  # Use log returns for normalization
        # norm_forecast = idc_forecast
        state = np.concatenate([state_info, norm_forecast, ])# idc_confidence])
        
        return np.array(state, dtype=np.float32)
    
    def get_idc_forecast(self, horizon_length: int = 4 * 24):
        return self.forecasters.idc.predict(horizon_length), np.ones(horizon_length) # Placeholder confidence levels

    def get_idc_trades(self):
        # Placeholder for retrieving current trades in the intraday market
        return self.idc.get_realized_trades()
    
    def _get_daa_market_state(self):


        next_day_prices = self.daa._get_hourly_prices_for_tomorrow()
        # next_day_prices = self.forecasters.daa.predict(24) # Forecast of the next 24 hours
        if next_day_prices is None or len(next_day_prices) == 0 or np.isnan(next_day_prices).any():
            next_day_prices = np.zeros(24)  # Default to zeros if no data available
        current_daa_promise = self._current_daa_promise[0]
        self._current_daa_promise = self._current_daa_promise[1:]  # Shift the promise buffer


        # Placeholder for retrieving the state of the day-ahead market
        return np.array([current_daa_promise] + list(next_day_prices), dtype=np.float32)

    def get_daa_current_promise(self):
        if self._current_daa_promise is None or len(self._current_daa_promise) == 0:
            if self.time_step > 100:
                # print(f"Warning: No current DAA promise available at time step {self.time_step}. Returning 0.")
                # print(self._current_daa_promise)
                return self._pending_daa_schedule[0]
            return 0.0  # Return 0 if there is no current promise
        return self._current_daa_promise[0]  # Return the current promised schedule for DAA

    def reset_prices(self, new_idc_prices, new_daa_prices):
        """
        Reset the market prices to new values based on the provided starting date.

        Args:
            new_idc_prices (list): The new IDC prices.
            new_daa_prices (list): The new DAA prices.
        """

        self.idc._price_profile_per_simulation_time_step = new_idc_prices
        if self.has_daa:
            self.daa._price_profile_per_simulation_time_step = new_daa_prices
            self.forecasters.daa._forecast_data = np.array(new_daa_prices)
        # self.idc_market = IdcMarket(price_profile_per_simulation_time_step=new_idc_prices)
        # self.idc_price_forcaster = DataProfileForecaster(new_idc_prices, time_delta_seconds=900)


        # self.daa_market = DaaMarket(price_profile_per_simulation_time_step=new_daa_prices)
        # self.daa_price_forcaster = DataProfileForecaster(forecast_data=new_daa_prices,
        #                                                  time_delta_seconds=3600)
        
        # self.forecasters = Forcasters(idc=self.idc_price_forcaster,
        #                               daa = self.daa_price_forcaster
        #                               )
        self.forecasters.idc._forecast_data = np.array(new_idc_prices)
        

        # Recalculate normalization bounds for the new week — critical for
        # reward() and _get_idc_market_state() to normalize correctly
        self.price_min = self.idc.data_profile.min()
        self.price_max = self.idc.data_profile.max()

        if self.has_daa:
            self.price_min_daa = self.price_min # self.daa.data_profile.min()
            self.price_max_daa = self.price_max # self.daa.data_profile.max()

        self.price_mean = self.idc.data_profile.mean()
        self.price_std = self.idc.data_profile.std()

        self.time_step = 0


    def _normalize_price(self, price, market_type='idc'):
        if market_type == 'idc':
            # return 2 * (price - self.price_min) / (self.price_max - self.price_min + 1e-8) - 1
            # return np.tanh(self.idc._price_profile_per_simulation_time_step.iloc[self.time_step] - self.idc._price_profile_per_simulation_time_step.iloc[self.time_step-1] ) # for price momentum, but here we normalize by mean and variance for stability
            return (price - self.price_mean) / (self.price_std + 1e-8)
        elif market_type == 'daa':
            # return 2 * (price - self.price_min_daa) / (self.price_max_daa - self.price_min_daa + 1e-8) - 1
            # return np.tanh(self.daa._price_profile_per_simulation_time_step.iloc[self.daa_time_step-1] - self.daa._price_profile_per_simulation_time_step.iloc[self.daa_time_step-2])  # for price momentum, but here we normalize by mean and variance for stability
            return (price - self.price_mean) / (self.price_std + 1e-8)
        else:
            raise ValueError("market_type must be either 'idc' or 'daa'")
        
    def _log_returns_aligned(self, prices, eps=1e-8):
        prices = np.asarray(prices, dtype=np.float64)
        r = np.zeros_like(prices)
        if prices is None:
            return []
        if prices.size < 2:
            return np.zeros_like(prices)
        r[1:] = np.tanh(prices[1:] - prices[:-1])
        return r