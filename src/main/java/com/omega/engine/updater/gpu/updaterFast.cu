#define ETA 1e-8f

__device__ inline float adamw_lerp(float start, float end, float weight) {
    return fma(weight, end, fma(-weight, start, start));
}

extern "C"
__global__ void adamw_multi_tensor_kernel(float **weights, const float **grads, float **ms, float **vs, const int *offsets,
        int tensor_count, int total_length, float learn_rate, float beta1, float beta2, float beta1_correction,
        float beta2_correction, float eps, float weight_decay) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= total_length) {
        return;
    }

    int left = 0;
    int right = tensor_count;
    while (left + 1 < right) {
        int mid = (left + right) >> 1;
        if (offsets[mid] <= index) {
            left = mid;
        } else {
            right = mid;
        }
    }

    int tensor_id = left;
    int local_index = index - offsets[tensor_id];
    float *weight = weights[tensor_id];
    const float *grad_ptr = grads[tensor_id];
    float *m_ptr = ms[tensor_id];
    float *v_ptr = vs[tensor_id];

    float grad = grad_ptr[local_index];
    float m = m_ptr[local_index];
    float v = v_ptr[local_index];

    m = adamw_lerp(grad, m, beta1);
    v = adamw_lerp(grad * grad, v, beta2);
    m_ptr[local_index] = m;
    v_ptr[local_index] = v;

    m /= beta1_correction;
    v /= beta2_correction;
    weight[local_index] -= learn_rate * (m / (sqrtf(v) + eps) + weight_decay * weight[local_index]);
}
