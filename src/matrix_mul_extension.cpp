// src/matrix_mul_extension.cpp

#include <torch/extension.h>
#include <vector>

// Declare the CUDA function (from matrix_mul.cu)
void matrixMul(float* C, const float* A, const float* B, int N);

// Python interface for matrix multiplication
torch::Tensor matrix_mul(torch::Tensor A, torch::Tensor B) {
    int N = A.size(0);
    auto C = torch::zeros({N, N}, torch::kFloat32).cuda();

    // Call CUDA kernel
    matrixMul(C.data_ptr<float>(), A.data_ptr<float>(), B.data_ptr<float>(), N);

    return C;
}

// Bind module for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("matrix_mul", &matrix_mul, "Matrix multiplication using CUDA");
}
