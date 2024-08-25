import torch
import matrix_mul_extension  # Your custom CUDA kernel

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create two random tensors
A = torch.randn(1024, 1024, device=device)
B = torch.randn(1024, 1024, device=device)

# Warm-up the GPU
for _ in range(10):
    C_custom = matrix_mul_extension.matrix_mul(A, B)
torch.cuda.synchronize()  # Ensure warm-up runs complete

# =============================
# Baseline: Measure Default PyTorch Kernel (Matrix Multiplication on GPU)
# =============================

# Synchronize GPU (to ensure accurate measurement)
torch.cuda.synchronize()

# Start time (GPU timer)
start_pytorch = torch.cuda.Event(enable_timing=True)
end_pytorch = torch.cuda.Event(enable_timing=True)

start_pytorch.record()

# Run default PyTorch matrix multiplication
C_pytorch = torch.matmul(A, B)

# End time (GPU timer)
end_pytorch.record()

# Wait for everything to finish
torch.cuda.synchronize()

# Calculate time for PyTorch's built-in matrix multiplication
elapsed_time_pytorch = start_pytorch.elapsed_time(end_pytorch)  # Time in milliseconds
print(f"Default PyTorch matrix multiplication time: {elapsed_time_pytorch:.6f} milliseconds")

# =============================
# Custom Kernel: Measure Your CUDA Kernel
# =============================

# Synchronize GPU (to ensure accurate measurement)
torch.cuda.synchronize()

# Start time (GPU timer)
start_custom = torch.cuda.Event(enable_timing=True)
end_custom = torch.cuda.Event(enable_timing=True)

start_custom.record()

# Run custom CUDA matrix multiplication
C_custom = matrix_mul_extension.matrix_mul(A, B)

# End time (GPU timer)
end_custom.record()

# Wait for everything to finish
torch.cuda.synchronize()

# Calculate time for custom CUDA matrix multiplication
elapsed_time_custom = start_custom.elapsed_time(end_custom)  # Time in milliseconds
print(f"Custom CUDA matrix multiplication time: {elapsed_time_custom:.6f} milliseconds")
