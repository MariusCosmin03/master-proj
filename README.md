# Energy Trading Environment with PPO

A flexible, multi-market energy trading environment built with Gymnasium, designed for training RL agents (SAC/PPO).

## Features

 **Multiple Market Support**: Day-ahead, Intraday
 **SAC Implementation**: Production-ready SAC agent
 **Expandable Design**: Easy to add custom markets or modify existing ones  
 **Rich Observations**: Prices, moving averages, positions, capital, time features  
 **Realistic Trading**: Position limits, phisical limits


## Installation

```bash
pip install -r requirements.txt
```

For the installation, access to nrgise codebase is also required. Momentarily this is not openly available, but we are working to open source it soon.

## Quick Start

### Training

**Train on Day-Ahead + Intraday markets (CPU):**
```bash
python train.py
```

This is trained on CPU since GPU do not provide substantial improvements for SAC, but running it on a gpu is also possible.

<!-- **!NOT CURRENTLY IMPLEMENTED! Train on GPU (if available):**
```bash
python train.py --device cuda
```

**Auto-detect GPU (recommended):**
```bash
python train.py --mode train --markets day_ahead intraday --n_episodes 1000 --device auto
```

**Train on all markets:**
```bash
python train.py --mode train --markets day_ahead intraday --n_episodes 2000 --device auto
```

**Test a trained model:**
```bash
python train.py --mode test --markets day_ahead intraday --model_path ppo_energy_trading_best.pt 
```
-->
## GPU Acceleration

The system can **automatically detects and uses CUDA GPUs** when available. But since the benefits are not large, the default device is CPU.


**!NOT YET IMPLEMENTED! Device Selection:**
- `--device auto` (default): Automatically uses GPU if available, falls back to CPU
- `--device cuda`: Forces GPU usage (fails if CUDA unavailable)
- `--device cpu`: Forces CPU usage

**Check GPU availability:**
```bash
python check_gpu.py
```

**Example with GPU info:**
```bash
==================================================
Device Information
==================================================
Using device: CUDA
GPU: NVIDIA GeForce RTX ...
CUDA Version: 12.18
Number of GPUs: 1
GPU Memory: 16.00 GB
==================================================
```


## Architecture

### Environment Structure

```
EnergyTradingEnv
├── Observation Space (continuous)
│   ├── Market prices (normalized)
│   ├── Short-term moving averages
│   ├── Long-term moving averages
│   ├── Current positions per market
│   ├── Available capital
│   └── Time features (normalized time, hour encoding)
│
└── Action Space (continuous, [-1, 1] per market)
    └── Actions represent buy/sell intensity
```

### Market Types

| Market | Time Horizon | Volatility | Transaction Cost |
|--------|-------------|------------|------------------|
| Day-Ahead | 24h | Low (5%) | 0.1% |
| Intraday | 4h | Medium (15%) | 0.2% |
| Real-Time | 1h | High (25%) | 0.3% |
| Ancillary Services | 12h | Medium-Low (8%) | 0.15% |
| Capacity | 168h (weekly) | Very Low (3%) | 0.05% |


## !NOT DONE FOR NOW! Command-Line Arguments

### Training Arguments

```bash
python train.py \
    --mode train \
    --markets day_ahead intraday \
    --initial_capital 100000 \
    --max_position 1000 \
    --episode_length 1000 \
    --n_episodes 1000 \
    --n_steps 2048 \
    --learning_rate 3e-4 \
    --gamma 0.99 \
    --clip_epsilon 0.2 \
    --device auto \
    --seed 42
```


## Observation Space Details

For an environment with `N` active markets:

**Observation Dimension**: `4*N + 4`

```
[market_1_price, market_1_ma_short, market_1_ma_long, market_1_position,
 market_2_price, market_2_ma_short, market_2_ma_long, market_2_position,
 ...,
 market_N_price, market_N_ma_short, market_N_ma_long, market_N_position,
 normalized_capital, normalized_time, hour_sin, hour_cos]
```

## Action Space Details

For an environment with `N` active markets:

**Action Dimension**: `N`

Each action in `[-1, 1]` represents:
- `-1.0`: Maximum sell (limited by battery power)
- `0.0`: Hold
- `+1.0`: Maximum buy (limited by battery power)

## Reward Function

```python
reward = profit_idc + profit_daa - idc_correction
where:
  profit_idc = idc_price_norm * idc_trade_volume
  profit_daa = daa_price_norm * daa_trade_volume
  idc_correction = idc_price_norm * (promised_volume - actual_traded_volume)
```

The IDC is used to correct trades that would not be possible due to battery limitations.

## PPO Hyperparameters
```
learning_rate_scheduler = LinearSchedule(
    start=3e-4,
    end=1e-5,
    end_fraction=0.9,
)

model = SAC(
    'MlpPolicy',
    vec_env,
    learning_rate=learning_rate_scheduler,
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
    seed=seed,
)
```
## Project Structure

```
.
|-- data_splits
    |-- daa             # Daa data split into weeks of train/validation/test
    |-- idc             # IDC data split into weeks of train/validation/test
|-- environment.py      # Environment implementation
|-- eval.py             # Testing script
|-- train.py            # Training script
|-- requirements.txt    # Dependencies
|-- README.md           # This file
|-- training_callback.py # Callback used to display relevant training statistics
```

## Monitoring Training

The training callback provides multiple relevant information during training, such as market specific rewards and revenue, amount of constraint violations and more.


## License

MIT License - Feel free to use and modify for your projects!

## Citation

If you use this environment in your research, please cite:

```bibtex
@software{energy_trading_ppo,
  title={Energy Trading Environment with SAC},
  author={Craciun Marius-Cosmin},
  year={2026},
  url={https://github.com/MariusCosmin03/master-proj/}
}
```
