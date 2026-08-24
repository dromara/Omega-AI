cmake_minimum_required(VERSION 3.22)

project(omega_cudnn_sdpa LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # SM86: RTX 3090. SM89: RTX 4060 and NVIDIA L40.
    set(CMAKE_CUDA_ARCHITECTURES 86 89)
endif()

find_package(CUDAToolkit REQUIRED)

# FindJNI on older CMake releases also requires AWT, which is absent from the
# headless JDKs normally installed on training servers. This library only uses
# the JNI headers.
find_path(JNI_INCLUDE_DIR
    NAMES jni.h
    HINTS "$ENV{JAVA_HOME}/include"
)
find_path(JNI_PLATFORM_INCLUDE_DIR
    NAMES jni_md.h
    HINTS
        "$ENV{JAVA_HOME}/include/linux"
        "$ENV{JAVA_HOME}/include/win32"
        "$ENV{JAVA_HOME}/include/darwin"
)

if(NOT JNI_INCLUDE_DIR OR NOT JNI_PLATFORM_INCLUDE_DIR)
    message(FATAL_ERROR "JNI headers were not found. Set JAVA_HOME to a JDK installation")
endif()

set(CUDNN_ROOT "" CACHE PATH "cuDNN installation root")
set(CUDNN_FRONTEND_INCLUDE_DIR "" CACHE PATH "Directory containing cudnn_frontend.h")

find_path(CUDNN_INCLUDE_DIR
    NAMES cudnn.h
    HINTS
        "${CUDNN_ROOT}/include"
        "$ENV{CUDNN_ROOT}/include"
        "$ENV{CUDNN_PATH}/include"
)

find_library(CUDNN_LIBRARY
    NAMES cudnn libcudnn
    HINTS
        "${CUDNN_ROOT}/lib"
        "${CUDNN_ROOT}/lib64"
        "$ENV{CUDNN_ROOT}/lib"
        "$ENV{CUDNN_ROOT}/lib64"
        "$ENV{CUDNN_PATH}/lib"
        "$ENV{CUDNN_PATH}/lib64"
)

if(NOT CUDNN_INCLUDE_DIR)
    message(FATAL_ERROR "cudnn.h was not found. Set -DCUDNN_ROOT=/path/to/cudnn")
endif()

if(NOT CUDNN_LIBRARY)
    message(FATAL_ERROR "libcudnn was not found. Set -DCUDNN_ROOT=/path/to/cudnn")
endif()

if(NOT EXISTS "${CUDNN_FRONTEND_INCLUDE_DIR}/cudnn_frontend.h")
    message(FATAL_ERROR
        "cudnn_frontend.h was not found. Set "
        "-DCUDNN_FRONTEND_INCLUDE_DIR=/path/to/cudnn-frontend/include")
endif()

add_library(omega_cudnn_sdpa SHARED omega_cudnn_sdpa_jni.cu)

target_include_directories(omega_cudnn_sdpa PRIVATE
    ${JNI_INCLUDE_DIR}
    ${JNI_PLATFORM_INCLUDE_DIR}
    ${CUDNN_INCLUDE_DIR}
    ${CUDNN_FRONTEND_INCLUDE_DIR}
)

target_link_libraries(omega_cudnn_sdpa PRIVATE
    CUDA::cudart
    ${CUDNN_LIBRARY}
)

target_compile_options(omega_cudnn_sdpa PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math --expt-relaxed-constexpr>
)

set_target_properties(omega_cudnn_sdpa PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_SEPARABLE_COMPILATION ON
    OUTPUT_NAME omega_cudnn_sdpa
)
