# Prebuilt cuDNN SDPA JNI library

`libomega_cudnn_sdpa.so` is a Linux x86_64 fat binary containing:

- SM86 for RTX 3090
- SM89 for RTX 4060 and NVIDIA L40

Runtime dependencies:

- CUDA 11.x (`libcudart.so.11.0`)
- cuDNN 8.9.x (`libcudnn.so.8`)
- A sufficiently recent NVIDIA driver for the installed CUDA runtime

SHA-256:

```text
1F30CCA70DCE2500FF7B6E0A17FE906D7D04241E7EED82E861FE921B637C6444
```

Load it with:

```bash
java \
  -Domega.cudnn.sdpa.library="$PWD/native/lib/linux-x86_64/cu11-cudnn8/libomega_cudnn_sdpa.so" \
  -cp "target/classes:..." \
  your.MainClass
```
