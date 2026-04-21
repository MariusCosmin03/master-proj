"""
Quick Demo Script - Energy Trading Environment

This script demonstrates a complete workflow:
1. Environment setup
2. Quick training run
3. Evaluation
4. Results visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from energy_trading_env import EnergyTradingEnv, create_sample_price_profiles
from train import MarketConfig, create_env, evaluate_agent


def quick_demo():
    """Run a quick demonstration of the environment."""
    
    print("=" * 70)
    print("MULTI-MARKET ENERGY TRADING - QUICK DEMO")
    print("=" * 70)
    
    # Step 1: Generate price data
    print("\n[1/6] Generating sample price profiles...")
    price_profiles = create_sample_price_profiles(days=7, time_delta_seconds=900)
    print(f"     ✓ Created {len(price_profiles)} market price profiles")
    print(f"     ✓ Data range: {price_profiles['daa'].index[0]} to {price_profiles['daa'].index[-1]}")
    
    # Step 2: Configure markets
    print("\n[2/6] Configuring markets...")
    
    # Try different configurations by uncommenting:
    
    # Configuration 1: All markets (default)
    # markets_config = MarketConfig.all_markets(price_profiles)
    markets_config = MarketConfig.daa_only(price_profiles)  # DAA only for cleaner demo
    config_name = "All Markets (DAA + FCR + IDC)"
    
    # Configuration 2: DAA only
    # markets_config = MarketConfig.daa_only(price_profiles)
    # config_name = "DAA Only"
    
    # Configuration 3: DAA + IDC
    # markets_config = MarketConfig.daa_and_idc(price_profiles)
    # config_name = "DAA + IDC"
    
    enabled_markets = [k for k, v in markets_config.items() if v.get('enabled', False)]
    print(f"     ✓ Configuration: {config_name}")
    print(f"     ✓ Enabled markets: {enabled_markets}")
    
    # Step 3: Create environment
    print("\n[3/6] Creating environment...")
    env = create_env(
        markets_config=markets_config,
        episode_length=96 * 2,  # 1 day at 15-min intervals
        storage_capacity=100.0,  # kWh
        storage_max_power=50.0   # kW
    )
    
    print(f"     ✓ Observation space: {env.observation_space.shape}")
    print(f"     ✓ Action space: {env.action_space.shape}")
    print(f"     ✓ Episode length: {env.episode_length} steps")
    
    # Validate environment
    print("\n[4/6] Validating environment...")
    check_env(env, warn=True)
    print("     ✓ Environment validation passed!")
    
    # Step 4: Test with random agent
    print("\n[5/6] Testing with random agent...")
    obs, info = env.reset(seed=42)
    episode_reward = 0
    soc_history = [info['initial_soc']]
    
    for step in range(96):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        soc_history.append(info['soc'])
        
        if terminated or truncated:
            break
    
    print(f"     ✓ Random agent completed episode")
    print(f"     ✓ Total reward: {episode_reward:.2f}")
    print(f"     ✓ Total revenue: €{info['total_revenue']:.2f}")
    print(f"     ✓ Final SOC: {info['soc']:.2%}")
    
    # Step 5: Quick PPO training
    print("\n[6/6] Training PPO agent (quick run)...")
    print("     Note: This is a short training run for demonstration.")
    print("     For better results, increase total_timesteps in train_ppo.py")
    
    model = PPO(
        'MlpPolicy',
        env,
        learning_rate=3e-4,
        n_steps=256,
        batch_size=64,
        n_epochs=5,
        verbose=0
    )
    
    # Train for short time (just for demo)
    model.learn(total_timesteps=5_000, progress_bar=True)
    print("     ✓ Training completed!")
    
    # Evaluate trained agent
    print("\n[7/6] Evaluating trained agent...")
    results = evaluate_agent(model, env, n_episodes=5, render=False)
    
    # Plot comparison
    print("\n[8/6] Creating visualization...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Random agent SOC
    axes[0].plot(soc_history, label='Random Agent', linewidth=2)
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('State of Charge')
    axes[0].set_title('Random Agent - SOC Evolution')
    axes[0].set_ylim([0, 1])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Trained agent results
    axes[1].bar(['Random\nAgent', 'Trained\nAgent (PPO)'], 
                [episode_reward, results['mean_reward']],
                color=['lightcoral', 'lightgreen'],
                edgecolor='black',
                linewidth=1.5)
    axes[1].set_ylabel('Average Reward')
    axes[1].set_title('Performance Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('demo_results.png', dpi=300, bbox_inches='tight')
    print("     ✓ Plot saved to 'demo_results.png'")
    plt.close()
    
    # Summary
    print("\n" + "=" * 70)
    print("DEMO SUMMARY")
    print("=" * 70)
    print(f"\nConfiguration: {config_name}")
    print(f"Enabled Markets: {enabled_markets}")
    print(f"\nRandom Agent Performance:")
    print(f"  - Reward: {episode_reward:.2f}")
    print(f"  - Revenue: €{info['total_revenue']:.2f}")
    print(f"\nTrained Agent Performance (5 episodes):")
    print(f"  - Mean Reward: {results['mean_reward']:.2f} (±{results['std_reward']:.2f})")
    print(f"  - Mean Revenue: €{results['mean_revenue']:.2f} (±€{results['std_revenue']:.2f})")
    
    improvement = ((results['mean_reward'] - episode_reward) / abs(episode_reward)) * 100
    print(f"\nImprovement: {improvement:+.1f}%")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("=" * 70)
    print("1. Run 'python train_ppo.py' for full training (50k+ timesteps)")
    print("2. Customize markets in train_ppo.py (enable/disable markets)")
    print("3. Try different scenarios with 'python examples.py'")
    print("4. Load your own price data (see README.md)")
    print("5. Tune hyperparameters for better performance")
    print("\nDemo completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    quick_demo()