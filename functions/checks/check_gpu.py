# Load gpu if available, otherwise load cpu
# check if gpu is available and return the gpu name 
def check_gpu_comprehensive():
    """
    Check GPU availability using both CuPy and PyTorch.
    """
    cupy_available = False
    pytorch_available = False
    
    # Check CuPy
    try:
        import cupy as cp
        if cp.cuda.runtime.getDeviceCount() > 0:
            props = cp.cuda.runtime.getDeviceProperties(0)
            gpu_name = props['name'].decode('utf-8') if isinstance(props['name'], bytes) else props['name']
            print(f"CuPy GPU detected: {gpu_name}")
            cupy_available = True
        else:
            print("CuPy: No GPU devices found")
    except Exception as e:
        print(f"CuPy: {e}")
    
    # Check PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            print(f"PyTorch GPU detected: {torch.cuda.get_device_name(0)}")
            pytorch_available = True
        else:
            print("PyTorch: No GPU available")
    except Exception as e:
        print(f"PyTorch: {e}")
    
    return cupy_available or pytorch_available

