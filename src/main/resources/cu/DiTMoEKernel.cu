extern "C"
__global__ void moe_route_top1_kernel(const float *logits, float *probs, float *top_idx, float *top_slot, float *top_weight, float *expert_counts, int N, int E) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) {
        return;
    }

    const float *row = logits + n * E;
    float max_v = row[0];
    for (int e = 1; e < E; ++e) {
        max_v = fmaxf(max_v, row[e]);
    }

    float sum = 0.0f;
    int best = 0;
    float best_p = -1.0f;
    for (int e = 0; e < E; ++e) {
        float p = expf(row[e] - max_v);
        probs[n * E + e] = p;
        sum += p;
    }
    sum = fmaxf(sum, 1.0e-20f);
    for (int e = 0; e < E; ++e) {
        float p = probs[n * E + e] / sum;
        probs[n * E + e] = p;
        if (p > best_p) {
            best_p = p;
            best = e;
        }
    }

    top_idx[n] = (float)best;
    top_weight[n] = best_p;
    float slot = atomicAdd(expert_counts + best, 1.0f);
    if (top_slot != top_weight) {
        top_slot[n] = slot;
    }
}

extern "C"
__global__ void moe_aux_loss_kernel(const float *probs, const float *expert_counts, float *aux_loss, int N, int E, float coef) {
    float sum = 0.0f;
    for (int e = 0; e < E; ++e) {
        float prob_sum = 0.0f;
        for (int n = 0; n < N; ++n) {
            prob_sum += probs[n * E + e];
        }
        float load = expert_counts[e] / (float)N;
        float prob_mean = prob_sum / (float)N;
        sum += load * prob_mean;
    }
    aux_loss[0] = (float)E * coef * sum;
}

extern "C"
__global__ void moe_combine_forward_kernel(const float *expert_out, float *out, const float *top_idx, const float *top_weight, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        out[idx] = expert_out[idx] * top_weight[n];
    }
}

extern "C"
__global__ void moe_dispatch_forward_kernel(const float *input, float *expert_input, const float *top_idx, const float *top_slot, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        int c = idx % C;
        int slot = (int)top_slot[n];
        expert_input[slot * C + c] = input[idx];
    }
}

extern "C"
__global__ void moe_combine_sparse_forward_kernel(const float *expert_out, float *out, const float *top_idx, const float *top_slot, const float *top_weight, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        int c = idx % C;
        int slot = (int)top_slot[n];
        out[idx] = expert_out[slot * C + c] * top_weight[n];
    }
}

extern "C"
__global__ void moe_expert_delta_kernel(const float *delta, const float *expert_out, float *expert_delta, float *top_weight_grad, const float *top_idx, const float *top_weight, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        float d = delta[idx];
        expert_delta[idx] = d * top_weight[n];
        atomicAdd(top_weight_grad + n, d * expert_out[idx]);
    } else {
        expert_delta[idx] = 0.0f;
    }
}

extern "C"
__global__ void moe_sparse_expert_delta_kernel(const float *delta, const float *expert_out, float *expert_delta, float *top_weight_grad, const float *top_idx, const float *top_slot, const float *top_weight, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        int c = idx % C;
        int slot = (int)top_slot[n];
        float d = delta[idx];
        expert_delta[slot * C + c] = d * top_weight[n];
        atomicAdd(top_weight_grad + n, d * expert_out[slot * C + c]);
    }
}

extern "C"
__global__ void moe_scatter_add_diff_kernel(const float *expert_diff, float *diff, const float *top_idx, const float *top_slot, int expert_id, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    int n = idx / C;
    if ((int)top_idx[n] == expert_id) {
        int c = idx % C;
        int slot = (int)top_slot[n];
        diff[idx] += expert_diff[slot * C + c];
    }
}

extern "C"
__global__ void moe_swiglu_forward_kernel(const float *w12_out, float *w1, float *w2, float *act, float *wt, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * H;
    if (idx >= len) {
        return;
    }
    int n = idx / H;
    int h = idx % H;
    float x1 = w12_out[n * (2 * H) + h];
    float x2 = w12_out[n * (2 * H) + H + h];
    float s = 1.0f / (1.0f + expf(-x1));
    float a = x1 * s;
    w1[idx] = x1;
    w2[idx] = x2;
    act[idx] = a;
    wt[idx] = a * x2;
}

extern "C"
__global__ void moe_swiglu_backward_kernel(const float *dwt, const float *w1, const float *w2, const float *act, float *w12_delta, int N, int H) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * H;
    if (idx >= len) {
        return;
    }
    int n = idx / H;
    int h = idx % H;
    float x1 = w1[idx];
    float x2 = w2[idx];
    float s = 1.0f / (1.0f + expf(-x1));
    float silu_grad = s * (1.0f + x1 * (1.0f - s));
    float d = dwt[idx];
    w12_delta[n * (2 * H) + h] = d * x2 * silu_grad;
    w12_delta[n * (2 * H) + H + h] = d * act[idx];
}

extern "C"
__global__ void moe_add_bias_kernel(float *output, const float *bias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * C;
    if (idx >= len) {
        return;
    }
    output[idx] += bias[idx % C];
}

extern "C"
__global__ void moe_bias_backward_kernel(const float *delta, float *diff_bias, int N, int C) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) {
        return;
    }
    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
        sum += delta[n * C + c];
    }
    diff_bias[c] = sum;
}

extern "C"
__global__ void moe_gate_backward_kernel(const float *probs, const float *top_idx, const float *top_weight_grad, const float *expert_counts, float *gate_delta, int N, int E, float coef) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = N * E;
    if (idx >= len) {
        return;
    }
    int n = idx / E;
    int e = idx % E;
    int top = (int)top_idx[n];

    float g_dot_p = 0.0f;
    for (int j = 0; j < E; ++j) {
        float g = ((j == top) ? top_weight_grad[n] : 0.0f) + (float)E * coef * (expert_counts[j] / (float)N) / (float)N;
        g_dot_p += g * probs[n * E + j];
    }

    float g_e = ((e == top) ? top_weight_grad[n] : 0.0f) + (float)E * coef * (expert_counts[e] / (float)N) / (float)N;
    gate_delta[idx] = probs[idx] * (g_e - g_dot_p);
}
