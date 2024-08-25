# CUDA-Optimized PyTorch Matrix Multiplication (CUDA_Custom_MatMul_Experiment)

This project integrates a custom CUDA-based matrix multiplication kernel into a PyTorch deep learning model, leveraging GPU acceleration for matrix operations. The goal is to compare the performance of this custom kernel with PyTorch's built-in matrix multiplication and demonstrate how custom CUDA kernels can optimize compute-intensive operations.

## Components Overview:

### 1. Custom CUDA Kernel for Matrix Multiplication (`matrix_mul.cu`)
- The CUDA kernel implements **tiled matrix multiplication**, dividing matrices into smaller tiles for efficient memory access.
- Each thread block computes one tile of the output matrix using **shared memory** to reduce global memory access latency.
- The kernel assumes both input matrices are square of size `N x N` and computes their product using `32 x 32` tiles to optimize performance.

### 2. C++ Interface for PyTorch (`matrix_mul_extension.cpp`)
- The custom kernel is exposed to PyTorch using C++ and **Pybind11**. This allows the kernel to be used in PyTorch like any other tensor operation.
- The function `matrix_mul` in the C++ file interfaces between PyTorch tensors and the CUDA kernel, handling data transfer between the CPU and GPU.

### 3. Convolutional Neural Network Models (`src/model.py`)
- **`CNN_CUSTOM`**: This model uses the custom CUDA kernel for matrix multiplication in parts of its forward pass. It mimics a basic convolutional neural network (CNN) used for image classification, using custom matrix multiplication for experimental purposes.
- **`CNN_STOCK`**: This model uses PyTorch's built-in `torch.matmul` for matrix multiplication, serving as the baseline for performance comparison with the custom kernel.

### 4. Training and Benchmarking (`train.py`, `train_benchmark.py`)
- **`train.py`**: This script trains the custom CNN model on the MNIST dataset, leveraging the custom CUDA kernel for specific operations.
- **`train_benchmark.py`**: This script compares the training times of the custom CNN (using the custom CUDA kernel) and the stock CNN (using PyTorch’s `torch.matmul`). It quantifies the performance gain from using a custom CUDA kernel.

### 5. Setup and Compilation (`setup.py`)
- The project is compiled using PyTorch’s extension mechanism, which allows CUDA code to be seamlessly integrated with Python. The `setup.py` handles building the C++/CUDA code as a Python extension module.

### 6. Testing Matrix Multiplication (`test.py`)
- This script tests the model’s accuracy and loss on the MNIST test set, validating whether the model works as expected after training with the custom kernel.

---

## Insights into Optimization

### 1. CUDA Tile-Based Optimization
- The matrix multiplication kernel divides matrices into tiles of size `32 x 32`, leveraging **shared memory** to reduce the overhead of accessing global memory. This improves memory bandwidth efficiency, crucial for performance in GPU-accelerated environments.
- Using shared memory and synchronizing threads within each block (`__syncthreads()`), the kernel accumulates partial results without redundant global memory accesses.

### 2. Comparison with PyTorch's `torch.matmul`
- PyTorch's matrix multiplication is highly optimized, leveraging **tensor cores** on supported hardware. The custom kernel, though optimized with shared memory, may not outperform PyTorch's `torch.matmul` due to these advanced optimizations.
- This project highlights the challenge of competing with highly optimized library functions like PyTorch's built-in operations. However, writing a custom kernel can still provide speedups for specialized operations or on hardware that may not fully utilize PyTorch’s optimizations.

### 3. Warm-Up and Synchronization
- Both PyTorch matrix multiplication and the custom kernel are warmed up before measuring performance, ensuring the GPU is fully initialized to reduce timing variability.
- The GPU is synchronized before and after each benchmark to ensure that all operations are completed before measuring time.

### 4. CUDA Optimization Challenges
- **Tile-Based Optimization**: While tile-based optimization is efficient for large square matrices, the current implementation assumes square matrices, limiting its flexibility for more complex models.
- **Tensor Core Utilization**: Further improvements could include exploiting **tensor cores** on modern GPUs like the A40, which provide specialized hardware for matrix multiplications, resulting in significant speedups.

---

## Future Work: Integrating Custom CUDA Kernel into the Training Loop

While the custom CUDA kernel is used experimentally in the forward pass, more meaningful integration would involve using the kernel to replace key operations during training:

1. **Fully Connected Layers**: The custom kernel could replace matrix multiplications in fully connected layers, where input sizes are large, and matrix multiplication is computationally expensive.

2. **Backpropagation Support**: Currently, the custom kernel does not support backpropagation via PyTorch’s autograd system. Implementing a **custom autograd function** would allow gradients to flow through the custom kernel, enabling it for use in training, not just inference.

3. **Handling Arbitrary Matrix Sizes**: The kernel could be extended to handle arbitrary matrix sizes by adapting the tiling approach for non-square matrices, making it more suitable for use in diverse architectures.

---

## Lesson/Project Outline for Others

This project provides valuable lessons in GPU programming, CUDA optimizations, and integrating custom operations with deep learning frameworks like PyTorch. Here's an outline for others to replicate the project:

1. **Basic Setup**: Follow the `setup.py` steps to compile and install the custom CUDA extension, introducing the process of building a PyTorch-compatible C++/CUDA module.

2. **Implement Tile-Based Matrix Multiplication**: Understand the tiled matrix multiplication technique used in the CUDA kernel. Experiment with changing tile sizes or exploring other optimizations (e.g., tensor cores).

3. **Benchmarking Custom Operations**: Use the `train_benchmark.py` script to compare the performance of your custom kernel with PyTorch's built-in operations. Analyze the impact of GPU hardware, tile size, and other factors on performance.

4. **Extend to Training**: Modify the `CNN_CUSTOM` model to fully integrate the custom kernel into the training process. This would require adding backward pass support via `torch.autograd.Function`.

5. **Explore Further Optimizations**: Investigate how to optimize the custom kernel by leveraging advanced CUDA features, such as warp-level programming, asynchronous memory transfers, or multi-GPU setups.

---

## Conclusion

This project demonstrates the integration of custom CUDA kernels in PyTorch models and explores the trade-offs between custom optimizations and highly optimized library functions like PyTorch’s `torch.matmul`. While custom kernels offer flexibility and the potential for optimization, they require careful design and benchmarking to compete with built-in alternatives.

Future work could extend the custom kernel to fully support training and exploit advanced GPU hardware features like tensor cores for further performance gains.

