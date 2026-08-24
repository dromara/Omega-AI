#include <jni.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <cudnn_frontend.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe = cudnn_frontend;

namespace {

constexpr int64_t Q_UID = 1;
constexpr int64_t K_UID = 2;
constexpr int64_t V_UID = 3;
constexpr int64_t O_UID = 4;
constexpr int64_t STATS_UID = 5;
constexpr int64_t SEQ_LEN_Q_UID = 6;
constexpr int64_t SEQ_LEN_KV_UID = 7;
constexpr int64_t DO_UID = 101;
constexpr int64_t DQ_UID = 102;
constexpr int64_t DK_UID = 103;
constexpr int64_t DV_UID = 104;

void throw_java(JNIEnv* env, const char* class_name, const std::string& message) {
    jclass exception_class = env->FindClass(class_name);
    if (exception_class != nullptr) {
        env->ThrowNew(exception_class, message.c_str());
    }
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void check_cudnn(cudnnStatus_t status, const char* operation) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + ": " + cudnnGetErrorString(status));
    }
}

void check_graph(fe::error_t status, const char* operation) {
    if (!status.is_good()) {
        throw std::runtime_error(std::string(operation) + ": " + status.get_message());
    }
}

void* pointer_from_jcuda(JNIEnv* env, jobject pointer) {
    if (pointer == nullptr) {
        throw std::invalid_argument("JCuda Pointer must not be null.");
    }
    jclass pointer_base = env->FindClass("jcuda/NativePointerObject");
    if (pointer_base == nullptr) {
        throw std::runtime_error("Unable to find jcuda.NativePointerObject.");
    }
    jfieldID field = env->GetFieldID(pointer_base, "nativePointer", "J");
    if (field == nullptr) {
        throw std::runtime_error("Unable to read JCuda nativePointer.");
    }
    return reinterpret_cast<void*>(static_cast<uintptr_t>(env->GetLongField(pointer, field)));
}

__global__ void pack_fp32_to_bf16_kernel(
    const float* input,
    __nv_bfloat16* output,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t padded_sequence,
    int64_t head_dim) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t length = batch * heads * padded_sequence * head_dim;
    if (index < length) {
        int64_t d = index % head_dim;
        int64_t s = (index / head_dim) % padded_sequence;
        int64_t bh = index / (padded_sequence * head_dim);
        if (s < sequence) {
            int64_t input_index = (bh * sequence + s) * head_dim + d;
            output[index] = __float2bfloat16_rn(input[input_index]);
        } else {
            output[index] = __float2bfloat16_rn(0.0f);
        }
    }
}

__global__ void unpack_bf16_to_fp32_kernel(
    const __nv_bfloat16* input,
    float* output,
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t padded_sequence,
    int64_t head_dim) {
    int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t length = batch * heads * sequence * head_dim;
    if (index < length) {
        int64_t d = index % head_dim;
        int64_t s = (index / head_dim) % sequence;
        int64_t bh = index / (sequence * head_dim);
        int64_t input_index = (bh * padded_sequence + s) * head_dim + d;
        output[index] = __bfloat162float(input[input_index]);
    }
}

void pack_fp32_to_bf16(const void* input, void* output, int64_t batch,
        int64_t heads, int64_t sequence, int64_t padded_sequence,
        int64_t head_dim, cudaStream_t stream) {
    constexpr int threads = 256;
    int64_t length = batch * heads * padded_sequence * head_dim;
    int blocks = static_cast<int>((length + threads - 1) / threads);
    pack_fp32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
        batch, heads, sequence, padded_sequence, head_dim);
    check_cuda(cudaGetLastError(), "pack_fp32_to_bf16_kernel");
}

void unpack_bf16_to_fp32(const void* input, void* output, int64_t batch,
        int64_t heads, int64_t sequence, int64_t padded_sequence,
        int64_t head_dim, cudaStream_t stream) {
    constexpr int threads = 256;
    int64_t length = batch * heads * sequence * head_dim;
    int blocks = static_cast<int>((length + threads - 1) / threads);
    unpack_bf16_to_fp32_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input), static_cast<float*>(output),
        batch, heads, sequence, padded_sequence, head_dim);
    check_cuda(cudaGetLastError(), "unpack_bf16_to_fp32_kernel");
}

std::shared_ptr<fe::graph::Graph> create_forward_graph(
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t head_dim,
    float scale,
    bool padding_mask) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto stride = std::vector<int64_t> {
        heads * sequence * head_dim,
        sequence * head_dim,
        head_dim,
        1
    };
    auto dims = std::vector<int64_t> {batch, heads, sequence, head_dim};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(Q_UID).set_dim(dims).set_stride(stride));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(K_UID).set_dim(dims).set_stride(stride));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(V_UID).set_dim(dims).set_stride(stride));

    auto options = fe::graph::SDPA_attributes()
        .set_name("omega_dit_sdpa_forward")
        .set_is_inference(false)
        .set_attn_scale(scale);

    if (padding_mask) {
        auto seq_len_q = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("SEQ_LEN_Q").set_uid(SEQ_LEN_Q_UID)
            .set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32));
        auto seq_len_kv = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("SEQ_LEN_KV").set_uid(SEQ_LEN_KV_UID)
            .set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32));
        options.set_padding_mask(true)
            .set_seq_len_q(seq_len_q)
            .set_seq_len_kv(seq_len_kv);
    }

    auto result = graph->sdpa(q, k, v, options);
    auto output = std::get<0>(result);
    auto stats = std::get<1>(result);
    output->set_output(true).set_uid(O_UID).set_dim(dims).set_stride(stride);
    stats->set_output(true).set_uid(STATS_UID).set_data_type(fe::DataType_t::FLOAT);
    return graph;
}

std::shared_ptr<fe::graph::Graph> create_backward_graph(
    int64_t batch,
    int64_t heads,
    int64_t sequence,
    int64_t head_dim,
    float scale,
    bool deterministic,
    bool padding_mask) {
    auto graph = std::make_shared<fe::graph::Graph>();
    graph->set_io_data_type(fe::DataType_t::BFLOAT16)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);

    auto dims = std::vector<int64_t> {batch, heads, sequence, head_dim};
    auto stride = std::vector<int64_t> {
        heads * sequence * head_dim,
        sequence * head_dim,
        head_dim,
        1
    };
    auto stats_dims = std::vector<int64_t> {batch, heads, sequence, 1};
    auto stats_stride = std::vector<int64_t> {heads * sequence, sequence, 1, 1};

    auto q = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Q").set_uid(Q_UID).set_dim(dims).set_stride(stride));
    auto k = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("K").set_uid(K_UID).set_dim(dims).set_stride(stride));
    auto v = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("V").set_uid(V_UID).set_dim(dims).set_stride(stride));
    auto output = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("O").set_uid(O_UID).set_dim(dims).set_stride(stride));
    auto d_output = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("dO").set_uid(DO_UID).set_dim(dims).set_stride(stride));
    auto stats = graph->tensor(fe::graph::Tensor_attributes()
        .set_name("Stats").set_uid(STATS_UID).set_dim(stats_dims)
        .set_stride(stats_stride).set_data_type(fe::DataType_t::FLOAT));

    auto options = fe::graph::SDPA_backward_attributes()
        .set_name("omega_dit_sdpa_backward")
        .set_attn_scale(scale)
        .set_deterministic_algorithm(deterministic);

    if (padding_mask) {
        auto seq_len_q = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("SEQ_LEN_Q").set_uid(SEQ_LEN_Q_UID)
            .set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32));
        auto seq_len_kv = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("SEQ_LEN_KV").set_uid(SEQ_LEN_KV_UID)
            .set_dim({batch, 1, 1, 1}).set_stride({1, 1, 1, 1})
            .set_data_type(fe::DataType_t::INT32));
        options.set_padding_mask(true)
            .set_seq_len_q(seq_len_q)
            .set_seq_len_kv(seq_len_kv);
    }

    auto result = graph->sdpa_backward(q, k, v, output, d_output, stats, options);
    auto d_q = std::get<0>(result);
    auto d_k = std::get<1>(result);
    auto d_v = std::get<2>(result);
    d_q->set_output(true).set_uid(DQ_UID).set_dim(dims).set_stride(stride);
    d_k->set_output(true).set_uid(DK_UID).set_dim(dims).set_stride(stride);
    d_v->set_output(true).set_uid(DV_UID).set_dim(dims).set_stride(stride);
    return graph;
}

struct SharedBackwardScratch {
    std::mutex mutex;
    void* d_output = nullptr;
    void* d_q = nullptr;
    void* d_k = nullptr;
    void* d_v = nullptr;
    void* workspace = nullptr;
    size_t tensor_bytes = 0;
    int64_t workspace_size = 0;

    ~SharedBackwardScratch() {
        cudaFree(workspace);
        cudaFree(d_v);
        cudaFree(d_k);
        cudaFree(d_q);
        cudaFree(d_output);
    }

    void ensure(size_t required_tensor_bytes, int64_t required_workspace_size) {
        if (required_tensor_bytes > tensor_bytes) {
            cudaFree(d_output);
            cudaFree(d_q);
            cudaFree(d_k);
            cudaFree(d_v);
            d_output = nullptr;
            d_q = nullptr;
            d_k = nullptr;
            d_v = nullptr;
            check_cuda(cudaMalloc(&d_output, required_tensor_bytes), "cudaMalloc shared dO");
            check_cuda(cudaMalloc(&d_q, required_tensor_bytes), "cudaMalloc shared dQ");
            check_cuda(cudaMalloc(&d_k, required_tensor_bytes), "cudaMalloc shared dK");
            check_cuda(cudaMalloc(&d_v, required_tensor_bytes), "cudaMalloc shared dV");
            tensor_bytes = required_tensor_bytes;
        }
        if (required_workspace_size > workspace_size) {
            cudaFree(workspace);
            workspace = nullptr;
            if (required_workspace_size > 0) {
                check_cuda(cudaMalloc(&workspace, static_cast<size_t>(required_workspace_size)),
                    "cudaMalloc shared workspace");
            }
            workspace_size = required_workspace_size;
        }
    }
};

SharedBackwardScratch g_backward_scratch;

struct SdpaPlan {
    int64_t batch;
    int64_t heads;
    int64_t sequence;
    int64_t padded_sequence;
    int64_t head_dim;
    int64_t padded_elements;
    bool padding_mask;

    cudnnHandle_t handle = nullptr;
    cudaStream_t stream = nullptr;
    std::shared_ptr<fe::graph::Graph> forward_graph;
    std::shared_ptr<fe::graph::Graph> backward_graph;

    void* q = nullptr;
    void* k = nullptr;
    void* v = nullptr;
    void* output = nullptr;
    void* stats = nullptr;
    void* seq_len_q = nullptr;
    void* seq_len_kv = nullptr;
    int64_t workspace_size = 0;

    SdpaPlan(int64_t batch_value,
             int64_t heads_value,
             int64_t sequence_value,
             int64_t head_dim_value,
             bool deterministic)
        : batch(batch_value),
          heads(heads_value),
          sequence(sequence_value),
          padded_sequence(cudnnGetVersion() < 8907 && sequence_value % 64 != 0
              ? ((sequence_value + 63) / 64) * 64
              : sequence_value),
          head_dim(head_dim_value),
          padded_elements(batch * heads * padded_sequence * head_dim),
          padding_mask(padded_sequence != sequence) {
        check_cudnn(cudnnCreate(&handle), "cudnnCreate");
        check_cudnn(cudnnSetStream(handle, stream), "cudnnSetStream");

        float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        forward_graph = create_forward_graph(
            batch, heads, padded_sequence, head_dim, scale, padding_mask);
        backward_graph = create_backward_graph(
            batch, heads, padded_sequence, head_dim, scale, deterministic, padding_mask);

        check_graph(forward_graph->build(handle, {fe::HeurMode_t::A}), "build SDPA forward");
        check_graph(backward_graph->build(handle, {fe::HeurMode_t::A}), "build SDPA backward");

        int64_t forward_workspace = 0;
        int64_t backward_workspace = 0;
        check_graph(forward_graph->get_workspace_size(forward_workspace), "get forward workspace");
        check_graph(backward_graph->get_workspace_size(backward_workspace), "get backward workspace");
        workspace_size = std::max(forward_workspace, backward_workspace);

        size_t tensor_bytes = static_cast<size_t>(padded_elements) * sizeof(__nv_bfloat16);
        size_t stats_bytes = static_cast<size_t>(batch * heads * padded_sequence) * sizeof(float);
        check_cuda(cudaMalloc(&q, tensor_bytes), "cudaMalloc Q");
        check_cuda(cudaMalloc(&k, tensor_bytes), "cudaMalloc K");
        check_cuda(cudaMalloc(&v, tensor_bytes), "cudaMalloc V");
        check_cuda(cudaMalloc(&output, tensor_bytes), "cudaMalloc O");
        check_cuda(cudaMalloc(&stats, stats_bytes), "cudaMalloc stats");
        if (padding_mask) {
            check_cuda(cudaMalloc(&seq_len_q, static_cast<size_t>(batch) * sizeof(int32_t)),
                "cudaMalloc SEQ_LEN_Q");
            check_cuda(cudaMalloc(&seq_len_kv, static_cast<size_t>(batch) * sizeof(int32_t)),
                "cudaMalloc SEQ_LEN_KV");
            std::vector<int32_t> sequence_lengths(static_cast<size_t>(batch),
                static_cast<int32_t>(sequence));
            check_cuda(cudaMemcpy(seq_len_q, sequence_lengths.data(),
                static_cast<size_t>(batch) * sizeof(int32_t), cudaMemcpyHostToDevice),
                "cudaMemcpy SEQ_LEN_Q");
            check_cuda(cudaMemcpy(seq_len_kv, sequence_lengths.data(),
                static_cast<size_t>(batch) * sizeof(int32_t), cudaMemcpyHostToDevice),
                "cudaMemcpy SEQ_LEN_KV");
        }
    }

    ~SdpaPlan() {
        cudaFree(seq_len_kv);
        cudaFree(seq_len_q);
        cudaFree(stats);
        cudaFree(output);
        cudaFree(v);
        cudaFree(k);
        cudaFree(q);
        if (handle != nullptr) {
            cudnnDestroy(handle);
        }
    }

    void forward(const void* q_fp32, const void* k_fp32, const void* v_fp32, void* output_fp32) {
        pack_fp32_to_bf16(q_fp32, q, batch, heads, sequence, padded_sequence,
            head_dim, stream);
        pack_fp32_to_bf16(k_fp32, k, batch, heads, sequence, padded_sequence,
            head_dim, stream);
        pack_fp32_to_bf16(v_fp32, v, batch, heads, sequence, padded_sequence,
            head_dim, stream);

        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, output}, {STATS_UID, stats}
        };
        if (padding_mask) {
            variant_pack[SEQ_LEN_Q_UID] = seq_len_q;
            variant_pack[SEQ_LEN_KV_UID] = seq_len_kv;
        }
        std::lock_guard<std::mutex> lock(g_backward_scratch.mutex);
        g_backward_scratch.ensure(0, workspace_size);
        check_graph(forward_graph->execute(handle, variant_pack, g_backward_scratch.workspace),
            "execute SDPA forward");
        unpack_bf16_to_fp32(output, output_fp32, batch, heads, sequence,
            padded_sequence, head_dim, stream);
    }

    void backward(const void* d_output_fp32, void* d_q_fp32, void* d_k_fp32, void* d_v_fp32) {
        size_t tensor_bytes = static_cast<size_t>(padded_elements) * sizeof(__nv_bfloat16);
        std::lock_guard<std::mutex> lock(g_backward_scratch.mutex);
        g_backward_scratch.ensure(tensor_bytes, workspace_size);

        pack_fp32_to_bf16(d_output_fp32, g_backward_scratch.d_output, batch, heads, sequence,
            padded_sequence, head_dim, stream);
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
            {Q_UID, q}, {K_UID, k}, {V_UID, v}, {O_UID, output},
            {DO_UID, g_backward_scratch.d_output}, {STATS_UID, stats},
            {DQ_UID, g_backward_scratch.d_q},
            {DK_UID, g_backward_scratch.d_k},
            {DV_UID, g_backward_scratch.d_v}
        };
        if (padding_mask) {
            variant_pack[SEQ_LEN_Q_UID] = seq_len_q;
            variant_pack[SEQ_LEN_KV_UID] = seq_len_kv;
        }
        check_graph(backward_graph->execute(handle, variant_pack, g_backward_scratch.workspace),
            "execute SDPA backward");
        unpack_bf16_to_fp32(g_backward_scratch.d_q, d_q_fp32, batch, heads, sequence,
            padded_sequence, head_dim, stream);
        unpack_bf16_to_fp32(g_backward_scratch.d_k, d_k_fp32, batch, heads, sequence,
            padded_sequence, head_dim, stream);
        unpack_bf16_to_fp32(g_backward_scratch.d_v, d_v_fp32, batch, heads, sequence,
            padded_sequence, head_dim, stream);
    }

    int64_t allocated_bytes() const {
        int64_t tensor_bytes = 4 * padded_elements *
            static_cast<int64_t>(sizeof(__nv_bfloat16));
        int64_t stats_bytes = batch * heads * padded_sequence *
            static_cast<int64_t>(sizeof(float));
        int64_t sequence_bytes = padding_mask
            ? 2 * batch * static_cast<int64_t>(sizeof(int32_t))
            : 0;
        return tensor_bytes + stats_bytes + sequence_bytes;
    }
};

SdpaPlan* plan_from_handle(jlong handle) {
    if (handle == 0) {
        throw std::invalid_argument("SDPA plan handle is null.");
    }
    return reinterpret_cast<SdpaPlan*>(static_cast<uintptr_t>(handle));
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeGetCudnnVersion(
    JNIEnv*, jclass) {
    return static_cast<jlong>(cudnnGetVersion());
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeGetAllocatedBytes(
    JNIEnv* env, jclass, jlong handle) {
    try {
        return static_cast<jlong>(plan_from_handle(handle)->allocated_bytes());
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT jlong JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeCreate(
    JNIEnv* env,
    jclass,
    jint batch,
    jint heads,
    jint sequence,
    jint head_dim,
    jboolean deterministic) {
    try {
        auto plan = std::make_unique<SdpaPlan>(
            batch, heads, sequence, head_dim, deterministic == JNI_TRUE);
        return static_cast<jlong>(reinterpret_cast<uintptr_t>(plan.release()));
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
        return 0;
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeForward(
    JNIEnv* env,
    jclass,
    jlong handle,
    jobject q,
    jobject k,
    jobject v,
    jobject output) {
    try {
        plan_from_handle(handle)->forward(
            pointer_from_jcuda(env, q),
            pointer_from_jcuda(env, k),
            pointer_from_jcuda(env, v),
            pointer_from_jcuda(env, output));
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeBackward(
    JNIEnv* env,
    jclass,
    jlong handle,
    jobject d_output,
    jobject d_q,
    jobject d_k,
    jobject d_v) {
    try {
        plan_from_handle(handle)->backward(
            pointer_from_jcuda(env, d_output),
            pointer_from_jcuda(env, d_q),
            pointer_from_jcuda(env, d_k),
            pointer_from_jcuda(env, d_v));
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
    }
}

extern "C" JNIEXPORT void JNICALL
Java_com_omega_engine_nn_layer_gpu_CudnnFlashAttentionKernel_nativeDestroy(
    JNIEnv* env,
    jclass,
    jlong handle) {
    try {
        delete plan_from_handle(handle);
    } catch (const std::exception& e) {
        throw_java(env, "java/lang/IllegalStateException", e.what());
    }
}
