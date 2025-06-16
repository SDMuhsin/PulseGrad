#include <cuda_runtime.h>
#include <cmath>

// CUDA kernel to perform the fused optimizer step
__global__ void experimental_fused_kernel(
    float* p_data,
    const float* grad,
    float* exp_avg,
    float* exp_avg_sq,
    float* previous_grad,
    int n_elements,
    float lr,
    float beta1,
    float beta2,
    float gamma,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    // Standard grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += blockDim.x * gridDim.x) {
        // Read data for the current element from global memory
        const float p_data_i = p_data[i];
        const float grad_i = grad[i];
        const float exp_avg_i = exp_avg[i];
        const float exp_avg_sq_i = exp_avg_sq[i];
        const float previous_grad_i = previous_grad[i];

        // --- Start of Fused Operations ---

        // 1. Apply weight decay if needed
        const float current_grad = grad_i + p_data_i * weight_decay;

        // 2. Update biased first moment estimate (m_t)
        const float new_exp_avg = exp_avg_i * beta1 + current_grad * (1.0f - beta1);

        // 3. Compute gradient difference
        const float diff = abs(previous_grad_i - current_grad);

        // 4. Update biased second moment estimate (v_t)
        const float new_val = current_grad * current_grad + gamma * diff * diff;
        const float new_exp_avg_sq = exp_avg_sq_i * beta2 + new_val * (1.0f - beta2);

        // 5. Compute bias-corrected moments (m_hat, v_hat)
        const float m_hat = new_exp_avg / bias_correction1;
        const float v_hat = new_exp_avg_sq / bias_correction2;

        // 6. Compute diffGrad friction coefficient (dfc)
        const float dfc = 1.0f / (1.0f + expf(-diff));

        // 7. Compute the final parameter update
        const float denom = sqrtf(v_hat) + eps;
        const float update = -lr * (m_hat * dfc) / denom;

        // --- End of Fused Operations ---

        // Write updated values back to global memory
        p_data[i] = p_data_i + update;
        exp_avg[i] = new_exp_avg;
        exp_avg_sq[i] = new_exp_avg_sq;
        previous_grad[i] = current_grad;
    }
}
