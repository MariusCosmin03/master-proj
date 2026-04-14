"""
training_callback.py
====================
Logs all market_info fields to TensorBoard during training.

Usage:
    from training_callback import TradingCallback

    callback = TradingCallback(log_freq=96, verbose=1)
    model.learn(total_timesteps=500_000, callback=callback)

Then inspect with:
    tensorboard --logdir ./logs
"""

from __future__ import annotations

import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback


class TradingCallback(BaseCallback):
    """
    Logs per-step and episodic metrics from the TradingEnvironment info dict.

    TensorBoard tags
    ─────────────────────────────────────────────────────────────────
    trading/idc_reward_mean          Mean IDC revenue per step (rolling window)
    trading/idc_reward_sum           Total IDC revenue (rolling window)
    trading/energy_penalty_mean      Mean constraint penalty per step
    trading/energy_penalty_sum       Total constraint penalty (rolling window)
    trading/net_reward_mean          Mean net reward (idc + penalty)
    trading/constraint_rate_pct      % of steps where a constraint violation fired
    trading/soc_mean                 Mean battery SoC
    trading/soc_std                  SoC standard deviation (how much of the battery is used)
    trading/soc_upper_violation_rate % of steps where SoC > 0.9
    trading/soc_lower_violation_rate % of steps where SoC < 0.1
    trading/price_mean               Mean IDC spot price
    trading/trade_mean               Mean signed trade volume (bias detector)
    trading/trade_abs_mean           Mean absolute trade volume (activity level)
    trading/charge_rate_pct          % of steps spent charging
    trading/discharge_rate_pct       % of steps spent discharging
    trading/idle_rate_pct            % of steps with no meaningful trade
    trading/power_balance_mean       Mean fixed power balance (grid imbalance)
    trading/power_balance_abs_mean   Mean absolute imbalance

    episode/total_idc_revenue        Total IDC revenue for completed episode
    episode/total_penalty            Total constraint penalty for completed episode
    episode/net_profit               Net profit for completed episode
    episode/n_constraint_violations  Count of constraint violations in episode
    episode/n_cycles                 Estimated battery cycles in episode
    episode/soc_upper_violations     SoC > 0.9 count in episode
    episode/soc_lower_violations     SoC < 0.1 count in episode
    episode/length                   Steps in completed episode
    """

    IDLE_THRESHOLD_MWH = 0.01

    def __init__(
        self,
        log_freq:          int   = 96,       # log every N steps (96 = 1 day)
        window:            int   = 960,      # rolling window size (960 = 10 days)
        battery_capacity_kwh: float = 1000.0,
        verbose:           int   = 0,
    ):
        super().__init__(verbose)
        self.log_freq         = log_freq
        self.window           = window
        self.battery_capacity_kwh = battery_capacity_kwh

        # ── Rolling buffers (capped at `window` steps) ───────────────────
        self._idc_reward     = deque(maxlen=window)
        self._energy_penalty = deque(maxlen=window)
        self._net_reward     = deque(maxlen=window)
        self._soc            = deque(maxlen=window)
        self._price          = deque(maxlen=window)
        self._trade          = deque(maxlen=window)
        self._power_balance  = deque(maxlen=window)
        self._constraint     = deque(maxlen=window)  # 0/1 per step

        # ── Episode accumulators (reset on each episode end) ─────────────
        self._ep_idc_revenue  = 0.0
        self._ep_penalty      = 0.0
        self._ep_violations   = 0
        self._ep_throughput   = 0.0
        self._ep_soc_upper    = 0
        self._ep_soc_lower    = 0
        self._ep_length       = 0

    # ─────────────────────────────────────────────────────────────────────
    def _on_step(self) -> bool:
        # self.locals["infos"] is a list — one per parallel env
        for info in self.locals["infos"]:
            self._ingest(info)

        if self.num_timesteps % self.log_freq == 0:
            self._log_rolling()

        return True  # False would stop training

    # ─────────────────────────────────────────────────────────────────────
    def _ingest(self, info: dict) -> None:
        """Push one step's info into all buffers and episode accumulators."""

        idc_r    = float(info.get("idc_reward",       0.0))
        energy_r = float(info.get("energy_reward",    0.0))
        soc      = float(info.get("soc",              0.5))
        price    = float(info.get("price",            0.0))
        trade    = float(info.get("trade",            0.0))
        balance  = float(info.get("fixed_power_balance", 0.0))
        c_fired  = int(info.get("constraint_fired",   0))

        # ── Rolling buffers ───────────────────────────────────────────────
        self._idc_reward.append(idc_r)
        self._energy_penalty.append(energy_r)
        self._net_reward.append(idc_r + energy_r)
        self._soc.append(soc)
        self._price.append(price)
        self._trade.append(trade)
        self._power_balance.append(balance)
        self._constraint.append(c_fired)

        # ── Episode accumulators ──────────────────────────────────────────
        self._ep_idc_revenue += idc_r
        self._ep_penalty     += energy_r
        self._ep_violations  += c_fired
        self._ep_throughput  += abs(trade)
        self._ep_soc_upper   += int(soc > 0.9)
        self._ep_soc_lower   += int(soc < 0.1)
        self._ep_length      += 1

        # ── Flush episode metrics on episode end ──────────────────────────
        if info.get("episode") or info.get("terminal_observation") is not None:
            self._log_episode()
            self._reset_episode()

    # ─────────────────────────────────────────────────────────────────────
    def _log_rolling(self) -> None:
        """Write rolling-window aggregates to TensorBoard."""

        trade_arr   = np.array(self._trade)
        soc_arr     = np.array(self._soc)
        balance_arr = np.array(self._power_balance)
        constraint_arr = np.array(self._constraint)

        charge_mask    = trade_arr < -self.IDLE_THRESHOLD_MWH
        discharge_mask = trade_arr >  self.IDLE_THRESHOLD_MWH
        idle_mask      = np.abs(trade_arr) <= self.IDLE_THRESHOLD_MWH
        n = len(trade_arr) + 1e-9

        # ── Revenue / penalty ─────────────────────────────────────────────
        self.logger.record("trading/idc_reward_mean",
                           float(np.mean(self._idc_reward)))
        self.logger.record("trading/idc_reward_sum",
                           float(np.sum(self._idc_reward)))
        self.logger.record("trading/energy_penalty_mean",
                           float(np.mean(self._energy_penalty)))
        self.logger.record("trading/energy_penalty_sum",
                           float(np.sum(self._energy_penalty)))
        self.logger.record("trading/net_reward_mean",
                           float(np.mean(self._net_reward)))

        # ── Constraint violations ─────────────────────────────────────────
        self.logger.record("trading/constraint_rate_pct",
                           float(constraint_arr.mean() * 100))

        # ── SoC ───────────────────────────────────────────────────────────
        self.logger.record("trading/soc_mean",   float(soc_arr.mean()))
        self.logger.record("trading/soc_std",    float(soc_arr.std()))
        self.logger.record("trading/soc_upper_violation_rate",
                           float((soc_arr > 0.9).mean() * 100))
        self.logger.record("trading/soc_lower_violation_rate",
                           float((soc_arr < 0.1).mean() * 100))

        # ── Price ─────────────────────────────────────────────────────────
        self.logger.record("trading/price_mean",
                           float(np.mean(self._price)))

        # ── Trade behaviour ───────────────────────────────────────────────
        self.logger.record("trading/trade_mean",
                           float(trade_arr.mean()))
        self.logger.record("trading/trade_abs_mean",
                           float(np.abs(trade_arr).mean()))
        self.logger.record("trading/charge_rate_pct",
                           float(charge_mask.sum() / n * 100))
        self.logger.record("trading/discharge_rate_pct",
                           float(discharge_mask.sum() / n * 100))
        self.logger.record("trading/idle_rate_pct",
                           float(idle_mask.sum() / n * 100))

        # ── Power balance ─────────────────────────────────────────────────
        self.logger.record("trading/power_balance_mean",
                           float(balance_arr.mean()))
        self.logger.record("trading/power_balance_abs_mean",
                           float(np.abs(balance_arr).mean()))

        if self.verbose > 0:
            self._print_rolling(trade_arr, soc_arr, constraint_arr)

    # ─────────────────────────────────────────────────────────────────────
    def _log_episode(self) -> None:
        """Write episode-level aggregates to TensorBoard."""
        cap_mwh = self.battery_capacity_kwh / 1000.0
        n_cycles = (self._ep_throughput / (2 * cap_mwh)) if cap_mwh > 0 else 0.0

        self.logger.record("episode/total_idc_revenue",       self._ep_idc_revenue)
        self.logger.record("episode/total_penalty",           self._ep_penalty)
        self.logger.record("episode/net_profit",              self._ep_idc_revenue + self._ep_penalty)
        self.logger.record("episode/n_constraint_violations", self._ep_violations)
        self.logger.record("episode/n_cycles",                n_cycles)
        self.logger.record("episode/soc_upper_violations",    self._ep_soc_upper)
        self.logger.record("episode/soc_lower_violations",    self._ep_soc_lower)
        self.logger.record("episode/length",                  self._ep_length)

        if self.verbose > 0:
            print(
                f"\n  ▸ Episode end  |  "
                f"IDC revenue: €{self._ep_idc_revenue:>10.2f}  |  "
                f"Penalty: €{self._ep_penalty:>10.2f}  |  "
                f"Net: €{self._ep_idc_revenue + self._ep_penalty:>10.2f}  |  "
                f"Violations: {self._ep_violations}  |  "
                f"Cycles: {n_cycles:.1f}"
            )

    # ─────────────────────────────────────────────────────────────────────
    def _reset_episode(self) -> None:
        self._ep_idc_revenue = 0.0
        self._ep_penalty     = 0.0
        self._ep_violations  = 0
        self._ep_throughput  = 0.0
        self._ep_soc_upper   = 0
        self._ep_soc_lower   = 0
        self._ep_length      = 0

    # ─────────────────────────────────────────────────────────────────────
    def _print_rolling(self, trade_arr, soc_arr, constraint_arr) -> None:
        n = len(trade_arr) + 1e-9
        charge_pct    = (trade_arr < -self.IDLE_THRESHOLD_MWH).sum() / n * 100
        discharge_pct = (trade_arr >  self.IDLE_THRESHOLD_MWH).sum() / n * 100
        idle_pct      = 100 - charge_pct - discharge_pct
        print(
            f"  [t={self.num_timesteps:>8d}]  "
            f"net_rew={float(np.mean(self._net_reward)):>8.2f}  |  "
            f"idc_rew={float(np.mean(self._idc_reward)):>8.2f}  |  "
            f"energy_pen={float(np.mean(self._energy_penalty)):>8.2f}  |  "
            f"SoC={soc_arr.mean():.2f}±{soc_arr.std():.2f}  |  "
            f"C/D/I={charge_pct:.0f}%/{discharge_pct:.0f}%/{idle_pct:.0f}%  |  "
            f"violations={constraint_arr.mean()*100:.1f}%"
        )
