#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>

// Include the kernel implementation directly.
// This replaces the forward declaration.
#include "experimental_cuda.cu"

// C++ function that will be called from Python
void experimental_fused_step(
    torch::Tensor& p,
    const torch::Tensor& grad,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& previous_grad,
    float lr,
    float beta1,
    float beta2,
    float gamma,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    // Input validation checks (remain the same)
    TORCH_CHECK(p.is_cuda(), "Parameter must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "Gradient must be a CUDA tensor");
    // ... (the rest of the function remains exactly the same) ...
    TORCH_CHECK(p.dtype() == torch::kFloat32, "All tensors must be float32");

    const int n_elements = p.numel();
    if (n_elements == 0) {
        return; // Nothing to do for empty tensors
    }

    // Configure the kernel launch
    const int threads_per_block = 256;
    const int blocks_per_grid = (n_elements + threads_per_block - 1) / threads_per_block;

    // Get the current CUDA stream in the modern way
    const auto stream = c10::cuda::getCurrentCUDAStream();

    // Launch the CUDA kernel
    experimental_fused_kernel<<<blocks_per_grid, threads_per_block, 0, stream>>>(
        p.data_ptr<float>(),
        grad.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        previous_grad.data_ptr<float>(),
        n_elements,
        lr,
        beta1,
        beta2,
        gamma,
        eps,
        weight_decay,
        bias_correction1,
        bias_correction2
    );
}

// Bind the C++ function to a Python module
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_step", &experimental_fused_step, "Experimental optimizer fused CUDA step");
}
