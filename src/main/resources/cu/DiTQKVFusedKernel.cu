extern "C"
__global__ void qkv_split_permute_kernel(const float *qkv, float *q, float *k, float *v, int B, int T, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = B * H * T * D;
    if (idx >= len) {
        return;
    }

    int d = idx % D;
    int t = (idx / D) % T;
    int h = (idx / D / T) % H;
    int b = idx / D / T / H;
    int embed = H * D;
    int base = (b * T + t) * 3 * embed + h * D + d;

    q[idx] = qkv[base];
    k[idx] = qkv[base + embed];
    v[idx] = qkv[base + 2 * embed];
}

extern "C"
__global__ void qkv_merge_unpermute_kernel(const float *dq, const float *dk, const float *dv, float *dqkv, int B, int T, int H, int D) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int len = B * H * T * D;
    if (idx >= len) {
        return;
    }

    int d = idx % D;
    int t = (idx / D) % T;
    int h = (idx / D / T) % H;
    int b = idx / D / T / H;
    int embed = H * D;
    int base = (b * T + t) * 3 * embed + h * D + d;

    dqkv[base] = dq[idx];
    dqkv[base + embed] = dk[idx];
    dqkv[base + 2 * embed] = dv[idx];
}
