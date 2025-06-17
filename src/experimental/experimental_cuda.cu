#include <cuda_runtime.h>
#include <cmath>

// Fused optimizer kernel with float4 vectorization to maximize memory bandwidth.
// This version correctly handles scalar-vector arithmetic by performing operations
// component-wise, resolving the compilation errors.
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
    // --- Vectorized Part (float4) ---
    // Process elements in chunks of 4 to increase memory throughput.
    // The grid-stride loop ensures all elements are processed, even with large tensors.
    const int num_vecs = n_elements / 4;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_vecs; i += gridDim.x * blockDim.x) {
        // Reinterpret pointers to handle float4 data types for coalesced memory access
        float4* p_data_vec = reinterpret_cast<float4*>(p_data);
        const float4* grad_vec = reinterpret_cast<const float4*>(grad);
        float4* exp_avg_vec = reinterpret_cast<float4*>(exp_avg);
        float4* exp_avg_sq_vec = reinterpret_cast<float4*>(exp_avg_sq);
        float4* previous_grad_vec = reinterpret_cast<float4*>(previous_grad);

        // --- Start of Fused Operations ---
        // Load 4 floats at once from global memory into registers
        const float4 p_data_i = p_data_vec[i];
        const float4 grad_i = grad_vec[i];
        const float4 exp_avg_i = exp_avg_vec[i];
        const float4 exp_avg_sq_i = exp_avg_sq_vec[i];
        const float4 previous_grad_i = previous_grad_vec[i];

        // 1. Apply weight decay if needed (component-wise)
        const float4 current_grad = make_float4(
            grad_i.x + p_data_i.x * weight_decay,
            grad_i.y + p_data_i.y * weight_decay,
            grad_i.z + p_data_i.z * weight_decay,
            grad_i.w + p_data_i.w * weight_decay
        );

        // 2. Update biased first moment estimate (m_t)
        const float one_minus_beta1 = 1.0f - beta1;
        const float4 new_exp_avg = make_float4(
            exp_avg_i.x * beta1 + current_grad.x * one_minus_beta1,
            exp_avg_i.y * beta1 + current_grad.y * one_minus_beta1,
            exp_avg_i.z * beta1 + current_grad.z * one_minus_beta1,
            exp_avg_i.w * beta1 + current_grad.w * one_minus_beta1
        );

        // 3. Compute gradient difference
        const float4 diff = make_float4(
            fabsf(previous_grad_i.x - current_grad.x),
            fabsf(previous_grad_i.y - current_grad.y),
            fabsf(previous_grad_i.z - current_grad.z),
            fabsf(previous_grad_i.w - current_grad.w)
        );

        // 4. Update biased second moment estimate (v_t)
        const float one_minus_beta2 = 1.0f - beta2;
        const float4 diff_sq = make_float4(diff.x * diff.x, diff.y * diff.y, diff.z * diff.z, diff.w * diff.w);
        const float4 grad_sq = make_float4(current_grad.x * current_grad.x, current_grad.y * current_grad.y, current_grad.z * current_grad.z, current_grad.w * current_grad.w);
        const float4 new_val = make_float4(
            grad_sq.x + gamma * diff_sq.x,
            grad_sq.y + gamma * diff_sq.y,
            grad_sq.z + gamma * diff_sq.z,
            grad_sq.w + gamma * diff_sq.w
        );
        const float4 new_exp_avg_sq = make_float4(
            exp_avg_sq_i.x * beta2 + new_val.x * one_minus_beta2,
            exp_avg_sq_i.y * beta2 + new_val.y * one_minus_beta2,
            exp_avg_sq_i.z * beta2 + new_val.z * one_minus_beta2,
            exp_avg_sq_i.w * beta2 + new_val.w * one_minus_beta2
        );

        // 5. Compute bias-corrected moments (m_hat, v_hat)
        const float4 m_hat = make_float4(new_exp_avg.x / bias_correction1, new_exp_avg.y / bias_correction1, new_exp_avg.z / bias_correction1, new_exp_avg.w / bias_correction1);
        const float4 v_hat = make_float4(new_exp_avg_sq.x / bias_correction2, new_exp_avg_sq.y / bias_correction2, new_exp_avg_sq.z / bias_correction2, new_exp_avg_sq.w / bias_correction2);

        // 6. Compute diffGrad friction coefficient (dfc)
        const float4 dfc = make_float4(1.0f / (1.0f + expf(-diff.x)), 1.0f / (1.0f + expf(-diff.y)), 1.0f / (1.0f + expf(-diff.z)), 1.0f / (1.0f + expf(-diff.w)));

        // 7. Compute the final parameter update
        const float4 denom = make_float4(sqrtf(v_hat.x) + eps, sqrtf(v_hat.y) + eps, sqrtf(v_hat.z) + eps, sqrtf(v_hat.w) + eps);
        const float4 update_numerator = make_float4(m_hat.x * dfc.x, m_hat.y * dfc.y, m_hat.z * dfc.z, m_hat.w * dfc.w);
        const float4 update = make_float4(
            -lr * (update_numerator.x / denom.x),
            -lr * (update_numerator.y / denom.y),
            -lr * (update_numerator.z / denom.z),
            -lr * (update_numerator.w / denom.w)
        );
        // --- End of Fused Operations ---

        // Write 4 floats at once back to global memory
        p_data_vec[i] = make_float4(p_data_i.x + update.x, p_data_i.y + update.y, p_data_i.z + update.z, p_data_i.w + update.w);
        exp_avg_vec[i] = new_exp_avg;
        exp_avg_sq_vec[i] = new_exp_avg_sq;
        previous_grad_vec[i] = current_grad;
    }

    // --- Scalar Part (Cleanup) ---
    // Handle remaining elements if n_elements is not a multiple of 4.
    // This loop ensures correctness for any tensor size.
    const int scalar_start_idx = num_vecs * 4;
    for (int i = scalar_start_idx + blockIdx.x * blockDim.x + threadIdx.x; i < n_elements; i += gridDim.x * blockDim.x) {
        const float p_data_i = p_data[i];
        const float grad_i = grad[i];
        const float exp_avg_i = exp_avg[i];
        const float exp_avg_sq_i = exp_avg_sq[i];
        const float previous_grad_i = previous_grad[i];

        // Standard scalar operations
        const float current_grad = grad_i + p_data_i * weight_decay;
        const float new_exp_avg = exp_avg_i * beta1 + current_grad * (1.0f - beta1);
        const float diff = fabsf(previous_grad_i - current_grad);
        const float new_val = current_grad * current_grad + gamma * diff * diff;
        const float new_exp_avg_sq = exp_avg_sq_i * beta2 + new_val * (1.0f - beta2);
        const float m_hat = new_exp_avg / bias_correction1;
        const float v_hat = new_exp_avg_sq / bias_correction2;
        const float dfc = 1.0f / (1.0f + expf(-diff));
        const float denom = sqrtf(v_hat) + eps;
        const float update = -lr * (m_hat * dfc) / denom;

        // Write updated values back to global memory
        p_data[i] = p_data_i + update;
        exp_avg[i] = new_exp_avg;
        exp_avg_sq[i] = new_exp_avg_sq;
        previous_grad[i] = current_grad;
    }
}

