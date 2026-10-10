#define BLOCK 1024

extern "C"
__global__ void vae_repa_smooth_l1(int n, const float *prediction,
        const float *target, float *loss, float beta) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    float x = prediction[id] - target[id];
    float ax = fabsf(x);
    loss[id] = ax < beta ? 0.5f * x * x / beta : ax - 0.5f * beta;
}

extern "C"
__global__ void vae_repa_smooth_l1_back(int n, const float *prediction,
        const float *target, float *diff, float beta) {
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= n) return;

    float x = prediction[id] - target[id];
    float ax = fabsf(x);
    float grad = ax < beta ? x / beta : (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f));
    diff[id] = grad / (float)n;
}
