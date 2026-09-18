# FA stability diagnostics

This is an **offline diagnostic**, not a change to the training algorithm. It
builds a separate native library and does not overwrite the production `.so`.
Never load the production and diagnostic libraries into the same JVM.

## What is checked

| Case | Purpose |
| --- | --- |
| quantization-only | Original input reference versus BF16-rounded input reference |
| FA-vs-original | Full change introduced by the FA boundary and implementation |
| FA-vs-BF16-input-reference | Error remaining with identical quantized Q/K/V/dO inputs |
| repeat | Same inputs, same plan, repeated forward/backward |
| private-scratch | Per-plan scratch versus production shared scratch |
| physical length | Same valid tokens, two different masked physical lengths |
| zero-repeat | Repeat a padded plan without poison to distinguish baseline instability |
| QKV-poison | Alternating +16/-16 in masked Q/K/V, valid data unchanged |
| dO-poison | Alternating +16/-16 in masked upstream gradients |
| scratch-growth | An earlier forward remains pending while other plans grow scratch |
| interleaved layers | All forwards, then reverse-order backwards without host synchronization between launches |

The zero-padding/shared-scratch baseline calls the actual `SdpaPlan::forward`
and `backward` compiled from `omega_cudnn_sdpa_jni.cu`. The diagnostic library
includes that source directly to avoid maintaining a second baseline algorithm.
Experimental poison/private-scratch paths reuse its graph builders, packing,
unpacking and workspace allocator; they only change the tested condition.

The independent reference computes noncausal attention and analytic gradients
on the CPU with double accumulation and stable softmax. It samples **complete
heads across the full original sequence**, not shorter attention problems.
Default sampled flattened `(batch, head)` indices are first, middle and last.
Increase `--reference-heads` to cover all B*H heads. GPU invariance tests always
check every element and head. The quantized reference is a mathematical
reference, not an exact emulation of cuDNN internal rounding or saved BF16 O.
No TF32, cuBLAS, Java attention caches or projection weights enter the reference.

## Build on the GPU server

Use the same CUDA, cuDNN and Frontend installations used for the training library.
The existing project prerequisites apply (JDK, CMake, CUDA, supported cuDNN).
Do not silently substitute another cuDNN version when diagnosing a version bug.

```bash
cmake -S native/cudnn_sdpa_demo -B build/fa-diagnostic \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCUDNN_ROOT=/path/to/cudnn \
  -DCUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn-frontend/include \
  -DOMEGA_SDPA_DIAGNOSTICS=ON
cmake --build build/fa-diagnostic --target omega_cudnn_sdpa_diagnostic -j

mkdir -p build/fa-diagnostic-java
javac -encoding UTF-8 -d build/fa-diagnostic-java \
  src/main/java/com/omega/example/dit/test/CudnnFlashAttentionStabilityDemo.java
java -cp build/fa-diagnostic-java \
  com.omega.example.dit.test.CudnnFlashAttentionStabilityDemo --self-test
```

This Java class needs no Omega/JCUDA dependency, no Maven build and no dataset.
`--self-test` does not load CUDA. It checks BF16 rounding, nonfinite detection,
zero-vector metrics, analytical gradients against finite differences, and the
snapshot format. It does not validate the GPU backend.

## Run

First run a smaller test with `--batch 1 --heads 2 --time 65 --dim 64`.
Then use the real training shape:

```bash
java -Xmx6g -cp build/fa-diagnostic-java \
  com.omega.example.dit.test.CudnnFlashAttentionStabilityDemo \
  --library "$PWD/build/fa-diagnostic/libomega_cudnn_sdpa_diagnostic.so" \
  --batch 12 --heads 12 --time 1101 --dim 64 \
  --reference-heads 3 --repeats 5 --layers 3 \
  --scales 1 --report fa-stability-scale1.csv
```

Repeat with `--scales 4,8` and a new report filename to stress peaked softmax.
`--padding-step 128 --layer-step 128` changes the physical-length comparison
and interleaved-layer increments from the default 64 to 128. This isolates
shape/alignment effects; it does not change production's automatic padding.
`--physical 256 --time 129`, for example, overrides the diagnostic baseline's
physical length while preserving 129 valid tokens. The default `--physical 0`
uses production's automatic padding. This option changes no production code.
Synthetic Q/K scales do not reproduce a particular trained model. Strong-scale
errors must be interpreted separately from typical training inputs.
CPU references take time; start with three heads, then increase coverage.
The default full-shape test retains several full input/output arrays; use a
large enough Java heap. GPU buffers are released when each plan closes.

The program prints the exact diagnostic library path/SHA256, GPU, cuDNN header
and runtime versions, valid/physical length, workspace size, and scratch mode.
This hash identifies the **diagnostic build**, not the production `.so` loaded
by another process. Confirm the production process's loaded library separately.

Every comparison prints O, dQ, dK and dV independently: max absolute error,
RMSE, relative L2, cosine, norm ratio, nonfinite counts, worst element with both
values, first four values, and worst head. CSV preserves summary metrics.
Input NaN/Inf aborts immediately rather than blaming FA for invalid inputs.

Default failure budgets are:

```text
reference:  --ref-rel 0.05
invariance: --invariant-rel 0.001
both:      --abs 0.00001
```

Each tensor AND each head must satisfy both:
`RMSE <= abs + rel * referenceRMS` and
`maxAbs <= abs + rel * referenceMaxAbs`. Any nonfinite value fails.
The quantization-only row is informational, not a pass/fail requirement.
These are diagnostic budgets, not universal numerical guarantees. Look at
absolute errors for near-zero gradients. A numerical FAIL exits with code 2;
setup/build/unsupported graph/runtime failures raise exceptions and are not
reported as PASS. Existing report/snapshot files are never overwritten.

`--deterministic true` is an optional separate experiment. The default is false,
matching production. If the installed backend cannot build a deterministic
graph, keep the failure log; the test does not silently fall back.

## Replay a real failing batch

The public Java writer takes contiguous FP32 **BHSD** arrays after QK norm and
RoPE. Capture Q, K, V and dO from the SAME layer and SAME forward/backward pair.
In `DiTAttentionLayer2.diff(cos, sin, igone)`, immediately before
`cudnn_sdpa.backward(temp, dqt, dkt, dvt)`, the relevant tensors are `rq`, `rk`,
`vt` and `temp`. They have not yet been overwritten by RoPE/projection backward.

An opt-in call you can temporarily insert at that point is:

```java
CudnnFlashAttentionStabilityDemo.writeSnapshot(
    java.nio.file.Paths.get("/omega/fa-step1300-layer5.ofa"),
    batchSize, headNum, time, dk,
    rq.syncHost().clone(), rk.syncHost().clone(),
    vt.syncHost().clone(), temp.syncHost().clone());
```

Import `com.omega.example.dit.test.CudnnFlashAttentionStabilityDemo`, guard this
call for the selected step/layer, and use a unique filename. Snapshotting is
synchronous and expensive; do NOT enable it unconditionally in training.
No capture hook is automatically added to production by this change.

```bash
java -Xmx6g -cp build/fa-diagnostic-java \
  com.omega.example.dit.test.CudnnFlashAttentionStabilityDemo \
  --library "$PWD/build/fa-diagnostic/libomega_cudnn_sdpa_diagnostic.so" \
  --snapshot /omega/fa-step1300-layer5.ofa \
  --reference-heads 12 --report fa-real-step1300-layer5.csv
```

Snapshot mode gets BHSD from the file and never rescales captured inputs.
`--write-snapshot inputs.ofa` saves synthetic inputs for cross-version tests.
Format: big-endian int32 magic `0x4f464131`, B,H,S,D; then four contiguous
big-endian float32 arrays Q,K,V,dO, each B*H*S*D elements. Exact file size is
validated before allocating arrays. A snapshot contains potentially sensitive
training activations; keep it local to the diagnostic environment.

## Interpret results

- Poison failures implicate masked padding or the diagnostic/backend path.
  Compare zero-poison and exact production baseline before drawing conclusions.
- Scratch/private/interleaving failures implicate ordering or saved-state reuse.
- Quantization-only error explains part of the precision change; FA-versus-
  quantized-reference still includes algorithmic rounding and BF16 outputs.
- If core tests pass but training diverges, capture real tensors from more layers
  and inspect QK norm/RoPE/projection backward, optimizer state and updates.
- These tests do not run a whole DiT, reconstruct the loss, validate JNI pointer
  offsets in the production Java wrapper, or prove long-term training stability.

No training process needs to be stopped or modified to run synthetic checks,
but avoid concurrent training when collecting reproducible diagnostic results.
