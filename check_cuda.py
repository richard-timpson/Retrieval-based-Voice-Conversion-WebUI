#!/usr/bin/env python3
"""
Script to check CUDA availability and GPU recognition in PyTorch
"""

import torch
import sys

def check_cuda_and_gpu():
    print("=" * 50)
    print("PyTorch CUDA and GPU Check")
    print("=" * 50)
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")
    
    if cuda_available:
        # Get CUDA version
        cuda_version = torch.version.cuda
        print(f"CUDA version: {cuda_version}")
        
        # Get cuDNN version
        cudnn_version = torch.backends.cudnn.version()
        print(f"cuDNN version: {cudnn_version}")
        
        # Check number of GPUs
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs available: {gpu_count}")
        
        # Get current GPU
        if gpu_count > 0:
            current_device = torch.cuda.current_device()
            print(f"Current GPU device: {current_device}")
            
            # List all GPUs
            print("\nGPU Details:")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # Convert to GB
                print(f"  GPU {i}: {gpu_name}")
                print(f"    Total memory: {gpu_memory:.2f} GB")
                
                # Check memory usage
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated(i) / 1024**3
                    cached = torch.cuda.memory_reserved(i) / 1024**3
                    print(f"    Memory allocated: {allocated:.2f} GB")
                    print(f"    Memory cached: {cached:.2f} GB")
        
        # Test GPU computation
        print("\nTesting GPU computation...")
        try:
            # Create a tensor on GPU
            x = torch.randn(1000, 1000).cuda()
            y = torch.randn(1000, 1000).cuda()
            z = torch.mm(x, y)
            print("✓ GPU computation test successful")
            
            # Check which device the tensor is on
            print(f"Tensor device: {z.device}")
            
            # Clean up
            del x, y, z
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"✗ GPU computation test failed: {e}")
    
    else:
        print("CUDA is not available. Possible reasons:")
        print("  - PyTorch was not installed with CUDA support")
        print("  - NVIDIA GPU drivers are not installed")
        print("  - CUDA toolkit is not installed")
        print("  - No compatible NVIDIA GPU found")
    
    print("\n" + "=" * 50)
    
    # Additional system info
    print("System Information:")
    print(f"Python version: {sys.version}")
    print(f"Platform: {sys.platform}")
    
    return cuda_available

if __name__ == "__main__":
    check_cuda_and_gpu() 