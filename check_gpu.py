#!/usr/bin/env python3
"""
GPU Verification Script
Checks if CUDA is available and displays GPU information
"""

import torch
import sys


def check_gpu():
    """Check GPU availability and display information"""
    
    print("\n" + "="*70)
    print("GPU VERIFICATION")
    print("="*70)
    
    # Check PyTorch installation
    print(f"\nPyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"cuDNN enabled: {torch.backends.cudnn.enabled}")
        
        # GPU information
        num_gpus = torch.cuda.device_count()
        print(f"\nNumber of GPUs: {num_gpus}")
        
        for i in range(num_gpus):
            print(f"\n--- GPU {i} ---")
            print(f"Name: {torch.cuda.get_device_name(i)}")
            
            props = torch.cuda.get_device_properties(i)
            print(f"Compute Capability: {props.major}.{props.minor}")
            print(f"Total Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"Multi-Processor Count: {props.multi_processor_count}")
            
            # Memory usage
            if torch.cuda.is_initialized():
                allocated = torch.cuda.memory_allocated(i) / 1e9
                reserved = torch.cuda.memory_reserved(i) / 1e9
                print(f"Memory Allocated: {allocated:.2f} GB")
                print(f"Memory Reserved: {reserved:.2f} GB")
        
        # Test GPU computation
        print("\n" + "="*70)
        print("TESTING GPU COMPUTATION")
        print("="*70)
        
        try:
            # Create tensors and move to GPU
            x = torch.randn(1000, 1000)
            y = torch.randn(1000, 1000)
            
            print("\nCPU computation...")
            import time
            start = time.time()
            z_cpu = torch.matmul(x, y)
            cpu_time = time.time() - start
            print(f"CPU time: {cpu_time*1000:.2f} ms")
            
            print("\nGPU computation...")
            x_gpu = x.cuda()
            y_gpu = y.cuda()
            torch.cuda.synchronize()
            start = time.time()
            z_gpu = torch.matmul(x_gpu, y_gpu)
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            print(f"GPU time: {gpu_time*1000:.2f} ms")
            
            speedup = cpu_time / gpu_time
            print(f"\nSpeedup: {speedup:.2f}x")
            
            # Verify results match
            z_gpu_cpu = z_gpu.cpu()
            max_diff = torch.max(torch.abs(z_cpu - z_gpu_cpu)).item()
            print(f"Max difference (CPU vs GPU): {max_diff:.2e}")
            
            if max_diff < 1e-4:
                print("✅ GPU computation verified successfully!")
            else:
                print("⚠️  Warning: GPU results differ from CPU")
                
        except Exception as e:
            print(f"Error during GPU test: {e}")
            return False
            
    else:
        print("\n CUDA is not available on this system.")
        print("\nPossible reasons:")
        print("1. PyTorch was installed without CUDA support")
        print("2. NVIDIA GPU drivers are not installed")
        print("3. No NVIDIA GPU is present")
        print("\nTo install PyTorch with CUDA support, visit:")
        print("https://pytorch.org/get-started/locally/")
        print("\nFor this project, training will use CPU (slower but still works)")
    
    print("\n" + "="*70)
    
    return cuda_available


def test_ppo_device():
    """Test PPO agent device placement"""
    print("\n" + "="*70)
    print("TESTING PPO AGENT DEVICE PLACEMENT")
    print("="*70)
    
    try:
        from ppo_agent import PPO
        
        # Test with auto device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"\nCreating PPO agent with device='{device}'...")
        
        agent = PPO(obs_dim=10, action_dim=2, device=device)
        
        # Check if model is on correct device
        model_device = next(agent.policy.parameters()).device
        print(f"Model device: {model_device}")
        
        # Test forward pass
        print("\nTesting forward pass...")
        obs = torch.randn(1, 10).to(device)
        with torch.no_grad():
            dist, value = agent.policy(obs)
        
        print(f"Distribution mean device: {dist.mean.device}")
        print(f"Value device: {value.device}")
        
        print("\nPPO agent device placement test passed!")
        return True
        
    except Exception as e:
        print(f"\nError during PPO test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("\n🔍 Starting GPU verification...\n")
    
    # Check GPU
    gpu_available = check_gpu()
    
    # Test PPO device placement
    ppo_ok = test_ppo_device()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if gpu_available:
        print("GPU is available and working")
        print(f"   Recommended: Use --device auto or --device cuda")
    else:
        print("GPU is not available")
        print(f"   Recommended: Use --device cpu (or --device auto)")
    
    if ppo_ok:
        print("PPO agent device handling is working correctly")
    else:
        print("PPO agent device handling failed")
    
    print("="*70 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if ppo_ok else 1)
