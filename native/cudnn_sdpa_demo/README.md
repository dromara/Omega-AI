# cuDNN SDPA comparison demo

For isolated padding, backward, scratch-reuse and real-tensor replay checks,
see [FA stability diagnostics](STABILITY_DIAGNOSTICS.md).

This native library uses NVIDIA cuDNN Frontend SDPA. It does not use any of
the FlashAttention CUDA kernels under `src/main`.

## Requirements

- NVIDIA GPU with compute capability 8.0 or newer.
- RTX 3090 uses SM86; RTX 4060 and NVIDIA L40 use SM89.
- CUDA Toolkit 11.8 or newer.
- cuDNN 8.9.6 or newer.
- A cuDNN Frontend release compatible with the installed cuDNN version.
- JDK 8 or newer and CMake 3.22 or newer.

The `nvidia-smi` CUDA version is the maximum CUDA version supported by the
driver. Check the installed toolkit separately with `nvcc --version`.

## Build for RTX 3090, RTX 4060, and L40

```bash
git clone --depth 1 --branch v1.9.0 \
  https://github.com/NVIDIA/cudnn-frontend.git \
  "$HOME/third_party/cudnn-frontend"

cmake -S native/cudnn_sdpa_demo \
  -B build/cudnn_sdpa_demo \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="86;89" \
  -DCUDNN_ROOT=/path/to/cudnn \
  -DCUDNN_FRONTEND_INCLUDE_DIR="$HOME/third_party/cudnn-frontend/include"

cmake --build build/cudnn_sdpa_demo -j
```

The output is `build/cudnn_sdpa_demo/libomega_cudnn_sdpa.so`.
It contains native code for RTX 3090 (SM86), RTX 4060 (SM89), and L40 (SM89).
The RTX 4060 does not need a separate binary from the L40.

A prebuilt CUDA 11/cuDNN 8 Linux binary is stored at:

```text
native/lib/linux-x86_64/cu11-cudnn8/libomega_cudnn_sdpa.so
```

## Run the comparison

Compile the Java project first, then run:

```bash
java \
  -Domega.cudnn.sdpa.library="$PWD/build/cudnn_sdpa_demo/libomega_cudnn_sdpa.so" \
  -cp "target/classes:$(cat classpath.txt)" \
  com.omega.example.dit.test.CudnnFlashAttentionCompareDemo
```

The default comparison shape matches the 256-pixel joint text/image sequence:

```text
B=2, H=12, T=333, D=64
```

It runs both `qkNorm=false` and `qkNorm=true`. Override dimensions or
tolerances when needed:

```bash
-Domega.sdpa.demo.batch=2
-Domega.sdpa.demo.time=333
-Domega.sdpa.demo.heads=12
-Domega.sdpa.demo.headDim=64
-Domega.sdpa.demo.maxAbs=0.02
-Domega.sdpa.demo.meanAbs=0.002
-Domega.sdpa.demo.gradMaxAbs=0.02
-Domega.sdpa.demo.gradMeanAbs=0.002
-Domega.sdpa.demo.warmup=10
-Domega.sdpa.demo.iterations=50
-Domega.sdpa.demo.qkNorm=both
```

The reference path remains FP32 while the cuDNN path uses BF16 inputs and FP32
softmax accumulation. Small numerical differences are therefore expected.
Timing runs two complete measured loops after warmup and reports only the second
loop. Memory output includes the reference core training tensors, the exact
native SDPA plan allocation, and the allocation delta observed by the CUDA
driver.

cuDNN releases below 8.9.7 require the key/value sequence dimension to be a
multiple of 64. The JNI library pads BHSD tensors internally and supplies
`padding_mask`, `SEQ_LEN_Q`, and `SEQ_LEN_KV` to cuDNN, then removes the padding
from the output and gradients. The Java tensor shape remains unchanged.
