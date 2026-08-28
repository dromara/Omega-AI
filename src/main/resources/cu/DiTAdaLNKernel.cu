extern "C"
__global__ void adaln_split6_kernel(
    const float* input,
    float* shift_msa,
    float* scale_msa,
    float* gate_msa,
    float* shift_mlp,
    float* scale_mlp,
    float* gate_mlp,
    int B,
    int D
) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    int len = B * D;

    if (idx >= len) {
        return;
    }

    int b = idx / D;
    int d = idx % D;
    int base = b * 6 * D + d;

    shift_msa[idx] = input[base];
    scale_msa[idx] = input[base + D];
    gate_msa[idx] = input[base + 2 * D];
    shift_mlp[idx] = input[base + 3 * D];
    scale_mlp[idx] = input[base + 4 * D];
    gate_mlp[idx] = input[base + 5 * D];
}

extern "C"
__global__ void adaln_set6_kernel(
    float* output,
    const float* d_shift_msa,
    const float* d_scale_msa,
    const float* d_gate_msa,
    const float* d_shift_mlp,
    const float* d_scale_mlp,
    const float* d_gate_mlp,
    int B,
    int D
) {
    int idx = (blockIdx.x + blockIdx.y * gridDim.x) * blockDim.x + threadIdx.x;
    int len = B * D;

    if (idx >= len) {
        return;
    }

    int b = idx / D;
    int d = idx % D;
    int base = b * 6 * D + d;

    output[base] = d_shift_msa[idx];
    output[base + D] = d_scale_msa[idx];
    output[base + 2 * D] = d_gate_msa[idx];
    output[base + 3 * D] = d_shift_mlp[idx];
    output[base + 4 * D] = d_scale_mlp[idx];
    output[base + 5 * D] = d_gate_mlp[idx];
}
