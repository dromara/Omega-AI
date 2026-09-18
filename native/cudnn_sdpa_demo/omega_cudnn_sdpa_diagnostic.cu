// Compile the actual production implementation into a SEPARATE diagnostic library.
// Do not load both libraries into the same JVM. No training entry point is changed.
#include "omega_cudnn_sdpa_jni.cu"
#include <sstream>

namespace {

__global__ void poison_padding(__nv_bfloat16* data, int64_t elements,
        int64_t valid, int64_t padded, int64_t dim, float magnitude) {
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < elements && (i / dim) % padded >= valid) {
        data[i] = __float2bfloat16_rn((i & 1) ? magnitude : -magnitude);
    }
}

struct DiagnosticPlan {
    std::unique_ptr<SdpaPlan> plan;
    SharedBackwardScratch private_scratch;
    float* tensors[8] = {}; // Q, K, V, dO, O, dQ, dK, dV; all external FP32.
    int64_t elements;
    bool independent;
    bool uploaded = false;
    bool forwarded = false;
    bool backwarded = false;

    DiagnosticPlan(int b, int h, int s, int d, int physical, bool isolated, bool deterministic)
        : elements(static_cast<int64_t>(b) * h * s * d), independent(isolated) {
        // physical=0 takes the production constructor and its version-dependent padding.
        plan = std::make_unique<SdpaPlan>(b, h, physical == 0 ? s : physical, d, deterministic);
        if (physical > s) {
            if (plan->padded_sequence != physical) {
                throw std::invalid_argument("Requested physical length was padded again; use a multiple of 64.");
            }
            // Buffers were allocated for the physical size. Rebuild with the real valid lengths.
            plan->sequence = s;
            plan->padding_mask = true;
            float scale = 1.0f / std::sqrt(static_cast<float>(d));
            plan->forward_graph = create_forward_graph(b, h, physical, d, scale, true);
            plan->backward_graph = create_backward_graph(b, h, physical, d, scale, deterministic, true);
            check_graph(plan->forward_graph->build(plan->handle, {fe::HeurMode_t::A}), "diagnostic forward build");
            check_graph(plan->backward_graph->build(plan->handle, {fe::HeurMode_t::A}), "diagnostic backward build");
            int64_t fw = 0, bw = 0;
            check_graph(plan->forward_graph->get_workspace_size(fw), "diagnostic forward workspace");
            check_graph(plan->backward_graph->get_workspace_size(bw), "diagnostic backward workspace");
            plan->workspace_size = std::max(fw, bw);
            check_cuda(cudaMalloc(&plan->seq_len_q, b * sizeof(int32_t)), "diagnostic seq Q");
            check_cuda(cudaMalloc(&plan->seq_len_kv, b * sizeof(int32_t)), "diagnostic seq KV");
            std::vector<int32_t> lengths(b, s);
            check_cuda(cudaMemcpy(plan->seq_len_q, lengths.data(), b * sizeof(int32_t), cudaMemcpyHostToDevice), "seq Q upload");
            check_cuda(cudaMemcpy(plan->seq_len_kv, lengths.data(), b * sizeof(int32_t), cudaMemcpyHostToDevice), "seq KV upload");
        }
        try {
            for (auto& tensor : tensors) {
                check_cuda(cudaMalloc(reinterpret_cast<void**>(&tensor), elements * sizeof(float)), "diagnostic FP32 buffer");
            }
        } catch (...) {
            for (auto tensor : tensors) cudaFree(tensor);
            throw;
        }
    }

    ~DiagnosticPlan() {
        cudaDeviceSynchronize();
        for (auto tensor : tensors) cudaFree(tensor);
    }

    void upload(JNIEnv* env, int index, jfloatArray values) {
        if (values == nullptr || env->GetArrayLength(values) != elements) {
            throw std::invalid_argument("Input array length differs from BHSD.");
        }
        jfloat* data = env->GetFloatArrayElements(values, nullptr);
        if (!data) throw std::runtime_error("Unable to access input array.");
        cudaError_t status = cudaMemcpy(tensors[index], data, elements * sizeof(float), cudaMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(values, data, JNI_ABORT);
        check_cuda(status, "diagnostic upload");
    }

    void poison(void* tensor, float value) {
        if (plan->padding_mask && value != 0.0f) {
            poison_padding<<<static_cast<int>((plan->padded_elements + 255) / 256), 256, 0, plan->stream>>>(
                static_cast<__nv_bfloat16*>(tensor), plan->padded_elements,
                plan->sequence, plan->padded_sequence, plan->head_dim, value);
            check_cuda(cudaGetLastError(), "poison padding");
        }
    }

    void forward(float pad_value) {
        if (!uploaded) throw std::logic_error("Upload inputs before forward.");
        forwarded = false;
        backwarded = false;
        if (!independent && pad_value == 0.0f) {
            // Baseline is the unmodified production path, including shared workspace.
            plan->forward(tensors[0], tensors[1], tensors[2], tensors[4]);
        } else {
            auto& p = *plan;
            void* destinations[] = {p.q, p.k, p.v};
            for (int i = 0; i < 3; ++i) {
                pack_fp32_to_bf16(tensors[i], destinations[i], p.batch, p.heads,
                    p.sequence, p.padded_sequence, p.head_dim, p.stream);
                poison(destinations[i], pad_value);
            }
            auto& scratch = independent ? private_scratch : g_backward_scratch;
            std::lock_guard<std::mutex> lock(scratch.mutex);
            scratch.ensure(0, p.workspace_size);
            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
                {Q_UID, p.q}, {K_UID, p.k}, {V_UID, p.v}, {O_UID, p.output}, {STATS_UID, p.stats}};
            if (p.padding_mask) {
                pack[SEQ_LEN_Q_UID] = p.seq_len_q;
                pack[SEQ_LEN_KV_UID] = p.seq_len_kv;
            }
            check_graph(p.forward_graph->execute(p.handle, pack, scratch.workspace), "diagnostic forward");
            unpack_bf16_to_fp32(p.output, tensors[4], p.batch, p.heads, p.sequence, p.padded_sequence, p.head_dim, p.stream);
        }
        forwarded = true;
    }

    void backward(float pad_value) {
        if (!forwarded) throw std::logic_error("Forward must precede backward.");
        if (!independent && pad_value == 0.0f) {
            plan->backward(tensors[3], tensors[5], tensors[6], tensors[7]);
        } else {
            auto& p = *plan;
            auto& scratch = independent ? private_scratch : g_backward_scratch;
            std::lock_guard<std::mutex> lock(scratch.mutex);
            scratch.ensure(p.padded_elements * sizeof(__nv_bfloat16), p.workspace_size);
            pack_fp32_to_bf16(tensors[3], scratch.d_output, p.batch, p.heads,
                p.sequence, p.padded_sequence, p.head_dim, p.stream);
            poison(scratch.d_output, pad_value);
            std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
                {Q_UID, p.q}, {K_UID, p.k}, {V_UID, p.v}, {O_UID, p.output}, {STATS_UID, p.stats},
                {DO_UID, scratch.d_output}, {DQ_UID, scratch.d_q}, {DK_UID, scratch.d_k}, {DV_UID, scratch.d_v}};
            if (p.padding_mask) {
                pack[SEQ_LEN_Q_UID] = p.seq_len_q;
                pack[SEQ_LEN_KV_UID] = p.seq_len_kv;
            }
            check_graph(p.backward_graph->execute(p.handle, pack, scratch.workspace), "diagnostic backward");
            void* gradients[] = {scratch.d_q, scratch.d_k, scratch.d_v};
            for (int i = 0; i < 3; ++i) {
                unpack_bf16_to_fp32(gradients[i], tensors[5 + i], p.batch, p.heads,
                    p.sequence, p.padded_sequence, p.head_dim, p.stream);
            }
        }
        backwarded = true;
    }
};

DiagnosticPlan& diagnostic(jlong handle) {
    if (!handle) throw std::invalid_argument("Diagnostic handle is null.");
    return *reinterpret_cast<DiagnosticPlan*>(static_cast<uintptr_t>(handle));
}

} // namespace

#define DIAG_METHOD(name) Java_com_omega_example_dit_test_CudnnFlashAttentionStabilityDemo_##name
#define DIAG_CATCH catch (const std::exception& e) { throw_java(env, "java/lang/IllegalStateException", e.what()); }

extern "C" JNIEXPORT jlong JNICALL DIAG_METHOD(nativeCreate)(JNIEnv* env, jclass,
        jint b, jint h, jint s, jint d, jint physical, jboolean isolated, jboolean deterministic) {
    try {
        if (b <= 0 || h <= 0 || s <= 0 || d <= 0 || (physical != 0 && physical < s)
                || static_cast<double>(b) * h * s * d > INT32_MAX) {
            throw std::invalid_argument("Invalid diagnostic shape.");
        }
        int device = 0;
        cudaDeviceProp prop;
        check_cuda(cudaGetDevice(&device), "get device");
        check_cuda(cudaGetDeviceProperties(&prop, device), "get device properties");
        if (prop.major < 8) throw std::runtime_error("FA diagnostics require an SM80+ GPU; this GPU is unsupported.");
        return reinterpret_cast<jlong>(new DiagnosticPlan(b, h, s, d, physical, isolated, deterministic));
    } DIAG_CATCH
    return 0;
}

extern "C" JNIEXPORT void JNICALL DIAG_METHOD(nativeUpload)(JNIEnv* env, jclass,
        jlong handle, jfloatArray q, jfloatArray k, jfloatArray v, jfloatArray dout) {
    try {
        auto& p = diagnostic(handle);
        p.uploaded = p.forwarded = p.backwarded = false;
        p.upload(env, 0, q); p.upload(env, 1, k); p.upload(env, 2, v); p.upload(env, 3, dout);
        p.uploaded = true;
    } DIAG_CATCH
}

extern "C" JNIEXPORT void JNICALL DIAG_METHOD(nativeForward)(JNIEnv* env, jclass, jlong handle, jfloat pad) {
    try { diagnostic(handle).forward(pad); } DIAG_CATCH
}

extern "C" JNIEXPORT void JNICALL DIAG_METHOD(nativeBackward)(JNIEnv* env, jclass, jlong handle, jfloat pad) {
    try { diagnostic(handle).backward(pad); } DIAG_CATCH
}

extern "C" JNIEXPORT void JNICALL DIAG_METHOD(nativeRead)(JNIEnv* env, jclass,
        jlong handle, jint index, jfloatArray result) {
    try {
        auto& p = diagnostic(handle);
        if (!p.backwarded || index < 0 || index > 3 || !result || env->GetArrayLength(result) != p.elements) {
            throw std::invalid_argument("Read requires completed forward/backward and a matching output array.");
        }
        check_cuda(cudaDeviceSynchronize(), "diagnostic synchronize");
        jfloat* data = env->GetFloatArrayElements(result, nullptr);
        if (!data) throw std::runtime_error("Unable to access result array.");
        cudaError_t status = cudaMemcpy(data, p.tensors[index + 4], p.elements * sizeof(float), cudaMemcpyDeviceToHost);
        env->ReleaseFloatArrayElements(result, data, 0);
        check_cuda(status, "diagnostic download");
    } DIAG_CATCH
}

extern "C" JNIEXPORT jstring JNICALL DIAG_METHOD(nativeDescribe)(JNIEnv* env, jclass, jlong handle) {
    try {
        cudaDeviceProp prop;
        int device = 0;
        check_cuda(cudaGetDevice(&device), "get device");
        check_cuda(cudaGetDeviceProperties(&prop, device), "get device properties");
        std::ostringstream out;
        out << "GPU=" << prop.name << " runtimeCuDNN=" << cudnnGetVersion()
            << " headerCuDNN=" << CUDNN_VERSION
            << " frontend=" << CUDNN_FRONTEND_MAJOR_VERSION << "." << CUDNN_FRONTEND_MINOR_VERSION
            << " compiledCUDA=" << __CUDACC_VER_MAJOR__ << "." << __CUDACC_VER_MINOR__;
        if (handle != 0) {
            auto& p = diagnostic(handle);
            out << " valid=" << p.plan->sequence << " physical=" << p.plan->padded_sequence
                << " padding=" << p.plan->padding_mask << " privateScratch=" << p.independent
                << " workspace=" << p.plan->workspace_size;
        }
        return env->NewStringUTF(out.str().c_str());
    } DIAG_CATCH
    return nullptr;
}

extern "C" JNIEXPORT void JNICALL DIAG_METHOD(nativeDestroy)(JNIEnv* env, jclass, jlong handle) {
    try { delete &diagnostic(handle); } DIAG_CATCH
}
