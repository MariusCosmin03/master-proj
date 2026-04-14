"""
evaluate.py
===========
Evaluation function for TradingEnvironment.

Usage:
    from evaluate import evaluate

    results = evaluate(env, model)          # single episode, prints + plots
    results = evaluate(env, model,
                       n_episodes=5,        # average over multiple runs
                       plot=True,
                       save_path="eval.png")
"""

from __future__ import annotations

import warnings
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from environment import TradingEnvironment
from train import MAX_EPISODE_STEPS, START_DATE, STORAGE_CAPACITY, STORAGE_POWER, TIME_DELTA_SECONDS, create_env, generate_prices

# ─────────────────────────────────────────────────────────────────────────────
# Core evaluation loop
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(
    env,
    model,
    n_episodes:     int  = 1,
    deterministic:  bool = True,
    plot:           bool = True,
    save_path:      Optional[str] = None,
    verbose:        bool = True,
) -> dict:
    """
    Roll out `model` in `env` for `n_episodes` and return a results dict.

    Collected per step
    ------------------
    idc_revenue        : price * realized_trade  (from info["idc_reward"])
    energy_penalty     : -10000 if battery couldn't honour the trade (from info["energy_reward"])
    soc                : battery state-of-charge  (obs[0])
    fixed_power_balance: net grid imbalance        (obs[1])
    price              : IDC spot price            (from markets.idc.prices_log)
    trade              : realized MWh trade        (from markets.idc.realized_trades_log)

    Computed metrics
    ----------------
    Financial
      total_revenue_eur        Total IDC revenue over episode
      total_penalty_eur        Sum of energy-constraint penalties
      net_profit_eur           revenue + penalty
      revenue_per_mwh          Avg €/MWh of throughput  (quality of dispatch)
      avg_charge_price         Mean price when charging
      avg_discharge_price      Mean price when discharging
      spread_captured_eur      avg_discharge - avg_charge  (actual arbitrage spread)
      max_possible_spread_eur  max(price) - min(price) in episode
      spread_efficiency_pct    spread_captured / max_possible_spread * 100

    Risk
      sharpe                   Annualised step-level Sharpe of IDC revenue
      sortino                  Annualised Sortino (downside only)
      max_drawdown_eur         Peak-to-trough drawdown of cumulative revenue
      revenue_volatility       Std of per-step revenue

    Trading behaviour
      total_throughput_mwh     Sum of |trade| across episode
      n_cycles                 throughput / (2 * battery_capacity_kwh / 1000)
      idle_steps_pct           % of steps where |trade| < threshold
      charge_steps             Count of charge steps
      discharge_steps          Count of discharge steps

    Battery / physical
      soc_mean                 Mean SoC
      soc_std                  Spread of SoC usage
      soc_upper_violations     Steps where SoC > 0.9
      soc_lower_violations     Steps where SoC < 0.1
      constraint_violations    Steps where energy_reward == -10000
      constraint_violation_pct % of steps with a constraint violation

    Timing quality
      price_timing_score       Pearson correlation: |trade| with price_rank * sign(trade)
                               +1 = perfect timing, 0 = random, -1 = inverted

    Suggested additions (not yet implemented — see NOTES at bottom)
      daily_revenue_std        Day-to-day revenue consistency
      forecast_mae             Forecast error vs realised prices  (needs forecaster access)
      degradation_cost_eur     Throughput × cost/MWh              (needs battery spec)
      net_profit_after_degradation
    """

    all_episodes: list[dict] = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False

        ep: dict = {
            "idc_revenue":     [],
            "energy_penalty":  [],
            "soc":             [],
            "power_balance":   [],
            "price":           [],
            "trade":           [],
        }

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # ── Pull raw values directly from market logs ──────────────────
            idc   = env.markets.idc
            ts    = idc.time_stamp
            price = float(idc.prices_log.get(ts, 0.0))
            trade = float(idc.realized_trades_log.get(ts, 0.0))

            ep["idc_revenue"].append(float(info.get("idc_reward", 0.0)))
            ep["energy_penalty"].append(float(info.get("energy_reward", 0.0)))
            ep["soc"].append(float(obs[0]))           # soc is obs[0] per _build_observation
            ep["power_balance"].append(float(obs[1])) # fixed_power_balance is obs[1]
            ep["price"].append(price)
            ep["trade"].append(trade)

        all_episodes.append({k: np.array(v) for k, v in ep.items()})

    # ── Average trajectories across episodes ──────────────────────────────
    def _avg(key): return np.mean([e[key] for e in all_episodes], axis=0)

    data = {k: _avg(k) for k in all_episodes[0]}

    # ── Compute metrics ───────────────────────────────────────────────────
    metrics = _compute_metrics(data, env)

    if verbose:
        _print_summary(metrics)

    if plot:
        fig = _plot(data, metrics, env)
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            if verbose:
                print(f"\n✓  Plot saved → {save_path}")
        else:
            plt.show()

    return {"metrics": metrics, "data": data}


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(data: dict, env) -> dict:
    rev      = data["idc_revenue"]
    penalty  = data["energy_penalty"]
    soc      = data["soc"]
    price    = data["price"]
    trade    = data["trade"]

    cap_kwh  = getattr(env, "battery_capacity_kwh", 1000.0)
    mwh_step = np.abs(trade)                            # MWh per step
    throughput = float(mwh_step.sum())

    charge_mask    = trade < -1e-6
    discharge_mask = trade >  1e-6

    avg_chg = float(price[charge_mask].mean())    if charge_mask.any()    else 0.0
    avg_dis = float(price[discharge_mask].mean()) if discharge_mask.any() else 0.0
    spread  = avg_dis - avg_chg
    max_spread = float(price.max() - price.min())

    # Risk metrics (annualised to yearly, assuming 15-min steps)
    steps_per_year = 96 * 365
    mean_r = rev.mean()
    std_r  = rev.std() + 1e-9
    sharpe = float(mean_r / std_r * np.sqrt(steps_per_year))

    down   = rev[rev < 0]
    sortino_denom = float(down.std() + 1e-9) if len(down) > 0 else 1e-9
    sortino = float(mean_r / sortino_denom * np.sqrt(steps_per_year))

    cumrev      = rev.cumsum()
    running_max = np.maximum.accumulate(cumrev)
    max_dd      = float((running_max - cumrev).max())

    # Price timing: positive when agent discharges at high prices, charges at low
    price_rank = pd.Series(price).rank(pct=True).values
    signed_rank = price_rank * np.sign(trade + 1e-12)
    timing = float(np.corrcoef(mwh_step, signed_rank)[0, 1]) if mwh_step.std() > 0 else 0.0

    idle_thresh = 0.01  # MWh — below this is considered idle
    idle_pct = float((mwh_step < idle_thresh).mean() * 100)

    violations  = (penalty < 0).sum()
    viol_pct    = float(violations / len(penalty) * 100)

    return {
        # ── Financial ──────────────────────────────────────────────────────
        "total_revenue_eur":        float(rev.sum()),
        "total_penalty_eur":        float(penalty.sum()),
        "net_profit_eur":           float(rev.sum() + penalty.sum()),
        "revenue_per_mwh":          float(rev.sum() / throughput) if throughput > 0 else 0.0,
        "avg_charge_price_eur_mwh": avg_chg,
        "avg_discharge_price_eur_mwh": avg_dis,
        "spread_captured_eur_mwh":  spread,
        "max_possible_spread_eur_mwh": max_spread,
        "spread_efficiency_pct":    float(spread / max_spread * 100) if max_spread > 0 else 0.0,

        # ── Risk ───────────────────────────────────────────────────────────
        "sharpe":                   sharpe,
        "sortino":                  sortino,
        "max_drawdown_eur":         max_dd,
        "revenue_volatility_eur":   float(std_r),

        # ── Trading behaviour ──────────────────────────────────────────────
        "total_throughput_mwh":     throughput,
        "n_cycles":                 float(throughput / (2 * cap_kwh / 1000)) if cap_kwh > 0 else 0.0,
        "idle_steps_pct":           idle_pct,
        "charge_steps":             int(charge_mask.sum()),
        "discharge_steps":          int(discharge_mask.sum()),

        # ── Battery / physical ─────────────────────────────────────────────
        "soc_mean":                 float(soc.mean()),
        "soc_std":                  float(soc.std()),
        "soc_upper_violations":     int((soc > 0.9).sum()),
        "soc_lower_violations":     int((soc < 0.1).sum()),
        "constraint_violations":    int(violations),
        "constraint_violation_pct": viol_pct,

        # ── Timing quality ─────────────────────────────────────────────────
        "price_timing_score":       timing,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Console output
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(m: dict):
    def _line(label, value, unit="", fmt=".2f"):
        print(f"  {label:<40s} {value:{fmt}} {unit}")

    print("\n" + "═" * 60)
    print("  EVALUATION SUMMARY")
    print("═" * 60)

    print("\n  ── Financial ───────────────────────────────────────────")
    _line("Total IDC Revenue",          m["total_revenue_eur"],           "€")
    _line("Total Constraint Penalties", m["total_penalty_eur"],           "€")
    _line("Net Profit",                 m["net_profit_eur"],              "€")
    _line("Revenue per MWh Throughput", m["revenue_per_mwh"],            "€/MWh")
    _line("Avg Charge Price",           m["avg_charge_price_eur_mwh"],   "€/MWh")
    _line("Avg Discharge Price",        m["avg_discharge_price_eur_mwh"],"€/MWh")
    _line("Spread Captured",            m["spread_captured_eur_mwh"],    "€/MWh")
    _line("Max Possible Spread",        m["max_possible_spread_eur_mwh"],"€/MWh")
    _line("Spread Efficiency",          m["spread_efficiency_pct"],      "%")

    print("\n  ── Risk ────────────────────────────────────────────────")
    _line("Sharpe Ratio (annualised)",  m["sharpe"],  "")
    _line("Sortino Ratio (annualised)", m["sortino"], "")
    _line("Max Drawdown",              m["max_drawdown_eur"],            "€")
    _line("Revenue Volatility (σ)",    m["revenue_volatility_eur"],      "€/step")

    print("\n  ── Trading Behaviour ───────────────────────────────────")
    _line("Total Throughput",          m["total_throughput_mwh"],        "MWh")
    _line("Battery Cycles",            m["n_cycles"],                    "cycles")
    _line("Idle Steps",                m["idle_steps_pct"],              "%")
    _line("Charge Steps",              m["charge_steps"],                "steps", ".0f")
    _line("Discharge Steps",           m["discharge_steps"],             "steps", ".0f")

    print("\n  ── Battery / Physical ──────────────────────────────────")
    _line("Mean SoC",                  m["soc_mean"],                    "")
    _line("SoC Std Dev",               m["soc_std"],                     "")
    _line("SoC Upper Violations (>90%)",m["soc_upper_violations"],       "steps", ".0f")
    _line("SoC Lower Violations (<10%)",m["soc_lower_violations"],       "steps", ".0f")
    _line("Constraint Violations",     m["constraint_violations"],       "steps", ".0f")
    _line("Constraint Violation Rate", m["constraint_violation_pct"],    "%")

    print("\n  ── Timing Quality ──────────────────────────────────────")
    score = m["price_timing_score"]
    label = ("excellent" if score > 0.6 else
             "good"      if score > 0.3 else
             "poor"      if score > 0.0 else "inverted")
    _line(f"Price Timing Score ({label})", score, "")

    print("\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_C = {
    "bg":       "#0D0D14",
    "panel":    "#13131C",
    "grid":     "#22223A",
    "text":     "#C8C8D8",
    "muted":    "#666680",
    "price":    "#F0C040",
    "revenue":  "#38D9A9",
    "penalty":  "#FF6B6B",
    "soc":      "#A78BFA",
    "charge":   "#60AFFE",
    "discharge":"#FF7043",
    "balance":  "#63E6BE",
    "accent":   "#F8F8FF",
}


def _ax_style(ax, title: str, xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(_C["panel"])
    ax.set_title(title, color=_C["text"], fontsize=10, fontweight="bold", pad=7)
    ax.set_xlabel(xlabel, color=_C["muted"], fontsize=8)
    ax.set_ylabel(ylabel, color=_C["muted"], fontsize=8)
    ax.tick_params(colors=_C["muted"], labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(_C["grid"])
    ax.yaxis.label.set_color(_C["muted"])
    ax.xaxis.label.set_color(_C["muted"])


def _plot(data: dict, metrics: dict, env) -> plt.Figure:
    t = np.arange(len(data["price"]))

    fig = plt.figure(figsize=(20, 18), facecolor=_C["bg"])
    fig.suptitle("TradingEnvironment — Evaluation Report",
                 color=_C["accent"], fontsize=16, fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(4, 2, figure=fig,
                           hspace=0.52, wspace=0.32,
                           left=0.07, right=0.96,
                           top=0.94, bottom=0.05)

    ax_price   = fig.add_subplot(gs[0, :])  # row 0 full width
    ax_cumrev  = fig.add_subplot(gs[1, 0])
    ax_soc     = fig.add_subplot(gs[1, 1])
    ax_trade   = fig.add_subplot(gs[2, 0])
    ax_balance = fig.add_subplot(gs[2, 1])
    ax_kpi     = fig.add_subplot(gs[3, 0])
    ax_daily   = fig.add_subplot(gs[3, 1])

    # ── 1. Price + trade markers ──────────────────────────────────────────
    _ax_style(ax_price, "IDC Price & Agent Trade Decisions",
              "Time Step (15 min)", "Price (€/MWh)")
    ax_price.plot(t, data["price"], color=_C["price"], lw=1, alpha=0.8, label="IDC Price")
    cm = data["trade"] < -1e-6
    dm = data["trade"] >  1e-6
    ax_price.scatter(t[cm], data["price"][cm], s=5, color=_C["charge"],    alpha=0.7, label="Charge",    zorder=3)
    ax_price.scatter(t[dm], data["price"][dm], s=5, color=_C["discharge"], alpha=0.7, label="Discharge", zorder=3)
    ax_price.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                    labelcolor=_C["text"], fontsize=8)

    # ── 2. Cumulative revenue vs penalties ────────────────────────────────
    _ax_style(ax_cumrev, "Cumulative Revenue & Penalties",
              "Time Step", "Cumulative (€)")
    cum_rev = np.cumsum(data["idc_revenue"])
    cum_pen = np.cumsum(data["energy_penalty"])
    cum_net = cum_rev + cum_pen
    ax_cumrev.plot(t, cum_rev, color=_C["revenue"],  lw=1.8, label="IDC Revenue")
    ax_cumrev.plot(t, cum_pen, color=_C["penalty"],  lw=1.2, ls="--", label="Constraint Penalties")
    ax_cumrev.plot(t, cum_net, color=_C["accent"],   lw=2.2, label="Net Profit")
    ax_cumrev.axhline(0, color=_C["grid"], lw=0.8, ls=":")
    ax_cumrev.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                     labelcolor=_C["text"], fontsize=8)
    ax_cumrev.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    # ── 3. SoC trajectory ─────────────────────────────────────────────────
    _ax_style(ax_soc, "Battery State of Charge", "Time Step", "SoC")
    ax_soc.plot(t, data["soc"], color=_C["soc"], lw=1.5)
    ax_soc.axhline(0.9, color=_C["penalty"], lw=1, ls="--", alpha=0.6, label="Limits (10%/90%)")
    ax_soc.axhline(0.1, color=_C["penalty"], lw=1, ls="--", alpha=0.6)
    ax_soc.fill_between(t, 0.1, 0.9, color=_C["soc"], alpha=0.04)
    ax_soc.set_ylim(-0.05, 1.05)
    # Highlight violations
    upper_viol = data["soc"] > 0.9
    lower_viol = data["soc"] < 0.1
    if upper_viol.any():
        ax_soc.scatter(t[upper_viol], data["soc"][upper_viol],
                       s=8, color=_C["penalty"], zorder=4, label="Violation")
    if lower_viol.any():
        ax_soc.scatter(t[lower_viol], data["soc"][lower_viol],
                       s=8, color=_C["penalty"], zorder=4)
    ax_soc.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                  labelcolor=_C["text"], fontsize=8)

    # ── 4. Trade volume per step ──────────────────────────────────────────
    _ax_style(ax_trade, "Trade Volume per Step",
              "Time Step", "Volume (MWh)")
    ax_trade.bar(t[data["trade"] < 0], data["trade"][data["trade"] < 0],
                 color=_C["charge"], alpha=0.7, width=1, label="Charge")
    ax_trade.bar(t[data["trade"] > 0], data["trade"][data["trade"] > 0],
                 color=_C["discharge"], alpha=0.7, width=1, label="Discharge")
    ax_trade.axhline(0, color=_C["grid"], lw=0.6)
    ax_trade.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                    labelcolor=_C["text"], fontsize=8)

    # ── 5. Power balance ─────────────────────────────────────────────────
    _ax_style(ax_balance, "Fixed Power Balance (Grid Imbalance)",
              "Time Step", "Power (kW)")
    ax_balance.plot(t, data["power_balance"], color=_C["balance"], lw=1, alpha=0.8)
    ax_balance.axhline(0, color=_C["grid"], lw=0.8, ls=":")
    ax_balance.fill_between(t, 0, data["power_balance"],
                             where=data["power_balance"] > 0,
                             color=_C["discharge"], alpha=0.15, label="Export")
    ax_balance.fill_between(t, 0, data["power_balance"],
                             where=data["power_balance"] < 0,
                             color=_C["charge"], alpha=0.15, label="Import")
    ax_balance.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                      labelcolor=_C["text"], fontsize=8)

    # ── 6. KPI bar chart ──────────────────────────────────────────────────
    _ax_style(ax_kpi, "Key Performance Indicators")
    kpi_labels = [
        "Net Profit\n(€)",
        "Spread\nEfficiency (%)",
        "Spread\nCaptured (€/MWh)",
        "Sharpe",
        "Timing\nScore",
        "SoC\nViolations",
    ]
    kpi_values = [
        metrics["net_profit_eur"],
        metrics["spread_efficiency_pct"],
        metrics["spread_captured_eur_mwh"],
        metrics["sharpe"],
        metrics["price_timing_score"] * 100,  # scale for visibility
        -(metrics["soc_upper_violations"] + metrics["soc_lower_violations"]),  # neg = bad
    ]
    bar_colors = [
        _C["revenue"] if v >= 0 else _C["penalty"] for v in kpi_values
    ]
    bars = ax_kpi.bar(kpi_labels, kpi_values, color=bar_colors,
                      edgecolor=_C["bg"], linewidth=1.2)
    for bar, val in zip(bars, kpi_values):
        ax_kpi.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (max(kpi_values) - min(kpi_values)) * 0.02,
            f"{val:.1f}",
            ha="center", va="bottom", color=_C["text"], fontsize=8
        )
    ax_kpi.axhline(0, color=_C["grid"], lw=0.8, ls=":")
    ax_kpi.tick_params(axis="x", labelsize=8)
    ax_kpi.set_facecolor(_C["panel"])
    for spine in ax_kpi.spines.values():
        spine.set_edgecolor(_C["grid"])
    ax_kpi.tick_params(colors=_C["muted"])

    # ── 7. Daily revenue breakdown ────────────────────────────────────────
    _ax_style(ax_daily, "Daily Revenue Breakdown",
              "Day", "Revenue (€)")
    steps_per_day = 96
    n_days = len(data["idc_revenue"]) // steps_per_day
    if n_days > 0:
        daily_rev = [
            data["idc_revenue"][d * steps_per_day:(d + 1) * steps_per_day].sum()
            for d in range(n_days)
        ]
        daily_pen = [
            data["energy_penalty"][d * steps_per_day:(d + 1) * steps_per_day].sum()
            for d in range(n_days)
        ]
        days = np.arange(n_days)
        ax_daily.bar(days, daily_rev, color=_C["revenue"], alpha=0.8, label="IDC Revenue", width=0.6)
        ax_daily.bar(days, daily_pen, color=_C["penalty"], alpha=0.8, label="Penalties",   width=0.6)
        ax_daily.axhline(0, color=_C["grid"], lw=0.8, ls=":")

        # Rolling 7-day average
        if n_days >= 7:
            roll = pd.Series(daily_rev).rolling(7, min_periods=1).mean().values
            ax_daily.plot(days, roll, color=_C["accent"], lw=1.8,
                          ls="--", label="7-day avg")

        ax_daily.legend(facecolor=_C["panel"], edgecolor=_C["grid"],
                        labelcolor=_C["text"], fontsize=8)
        ax_daily.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"€{x:,.0f}"))

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# NOTES: Suggested additional metrics
# ─────────────────────────────────────────────────────────────────────────────
#
# These are not implemented because they require data not directly available
# from the environment's step() output, but are worth adding:
#
# 1. DEGRADATION-ADJUSTED PROFIT
#    net_profit_after_degradation = net_profit - (throughput_mwh * degradation_cost_per_mwh)
#    Requires: battery replacement cost and cycle-life spec from battery model.
#    Why: the agent can appear profitable while destroying the battery asset.
#
# 2. FORECAST QUALITY (MAE / correlation)
#    Compare env.markets.forecasters.idc predictions against realised prices.
#    Why: distinguishes poor profits caused by bad forecasts vs bad policy.
#
# 3. DAILY REVENUE CONSISTENCY (std of daily revenue)
#    A stable strategy with moderate daily revenue is often preferable
#    to a high-mean, high-variance one in real operations.
#
# 4. OPPORTUNITY COST
#    (max_possible_spread - spread_captured) * throughput_mwh
#    How much money was left on the table relative to a perfect oracle.
#
# 5. SCHEDULE ADHERENCE RATE
#    Compare committed schedule (get_cleared_schedule) vs realised trades.
#    Why: measures whether the forward bids the agent places are actually
#    executed, which matters for real IDC market participation.
#
# 6. PENALTY TIMING
#    Which hours of day do constraint violations cluster in?
#    Early-morning/late-night violations suggest the agent doesn't handle
#    low-liquidity periods correctly.
#
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    
    import os
    from stable_baselines3 import PPO                  # swap for SAC, TD3, etc.
    from nrgise import EnergySystem
    from markets.idc_market import IdcMarket
    from markets_wrapper import MarketsWrapper
        
    from nrgise import EnergySystem, Grid, StorageSystemEnergyReservoir
    from nrgise.forecaster import DataProfileForecaster


    # ── 1. Load price data ────────────────────────────────────────────────
    # price_profile = pd.read_csv("data/eval_prices.csv",
    #                             index_col=0, parse_dates=True).squeeze()

    # ── 2. Build the market and energy system ─────────────────────────────
    print("\n1. Generating price profiles...")
    # price_profiles = create_sample_price_profiles(days=30, time_delta_seconds=900)
    timestamps = pd.date_range(start=START_DATE, periods=MAX_EPISODE_STEPS, freq='15T')
    
    idc_price_profile = generate_prices(base_price=80, price_volatility=0.05, max_steps=MAX_EPISODE_STEPS)

    # Configure markets
    print("\n2. Configuring markets...")
    
    # Choose one of these configurations:
    
    idc_market = IdcMarket(price_profile_per_simulation_time_step=pd.Series(idc_price_profile, index=timestamps))
    idc_price_forcaster = DataProfileForecaster(idc_price_profile, time_delta_seconds=900)
    markets = MarketsWrapper(
        initial_balance=10000.0,
        battery_capacity_kwh=STORAGE_CAPACITY,
        battery_max_power_kwh=STORAGE_POWER,
        intraday_market=idc_market,
        idc_price_forcaster=idc_price_forcaster,  # Placeholder, can be set to actual forecaster instance,
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
   
    
    
    
    # Create environment
    print("\n3. Creating environment...")
    env = create_env(
        energy_system=energy_system,
        markets=markets,
        forecast_horizon_hours=3,
        time_delta_seconds = TIME_DELTA_SECONDS,
        battery_power_kwh=STORAGE_POWER,
        battery_capacity_kwh=STORAGE_CAPACITY,
        max_episode_steps=MAX_EPISODE_STEPS,  # 24 hours with 15 min steps
        start_date=START_DATE,
    )

    # ── 4. Load the trained model ─────────────────────────────────────────
    MODEL_PATH = "models/idc_only/best/best_model"             # no .zip extension needed
    if not os.path.exists(MODEL_PATH + ".zip"):
        raise FileNotFoundError(f"No model found at {MODEL_PATH}.zip")

    model = PPO.load(MODEL_PATH, env=env)
    print(f"✓  Loaded model from {MODEL_PATH}")

    # ── 5. Run evaluation ─────────────────────────────────────────────────
    results = evaluate(
        env=env,
        model=model,
        n_episodes=1,           # increase to average over multiple runs
        deterministic=True,     # always True for eval
        plot=True,
        save_path="eval_report.png",
        verbose=True,
    )