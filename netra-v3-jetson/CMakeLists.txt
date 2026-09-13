cmake_minimum_required(VERSION 3.18)

# Detect CUDA before declaring languages
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    project(netra-v3 LANGUAGES C CXX CUDA)
    set(NETRA_HAS_CUDA_LANG TRUE)
    set(CMAKE_CUDA_ARCHITECTURES 87)
else()
    project(netra-v3 LANGUAGES C CXX)
    set(NETRA_HAS_CUDA_LANG FALSE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(NETRA_USE_CUDA    "Enable CUDA acceleration" ${NETRA_HAS_CUDA_LANG})
option(NETRA_BUILD_ASR   "Build whisper.cpp ASR"     ON)
option(NETRA_BUILD_TTS   "Build Piper TTS"           ON)

# ──────────────────────────────────────────────
# Third-party: llama.cpp
# ──────────────────────────────────────────────
if(NETRA_USE_CUDA)
    set(GGML_CUDA ON CACHE BOOL "ggml CUDA" FORCE)
else()
    set(GGML_CUDA OFF CACHE BOOL "ggml CUDA" FORCE)
endif()
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_APP OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/llama.cpp EXCLUDE_FROM_ALL)

# ──────────────────────────────────────────────
# Third-party: whisper.cpp
# ──────────────────────────────────────────────
if(NETRA_BUILD_ASR)
    if(NETRA_USE_CUDA)
        set(GGML_CUDA ON CACHE BOOL "ggml CUDA for whisper" FORCE)
    endif()
    set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)
endif()

# ──────────────────────────────────────────────
# Third-party: httplib (header-only)
# ──────────────────────────────────────────────
add_library(httplib INTERFACE)
target_include_directories(httplib INTERFACE third_party/cpp-httplib)

# ──────────────────────────────────────────────
# Third-party: nlohmann/json (header-only)
# ──────────────────────────────────────────────
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE third_party/json/single_include)

# ──────────────────────────────────────────────
# Find system libraries
# ──────────────────────────────────────────────
find_package(Threads REQUIRED)

# FAISS (optional)
find_library(FAISS_LIB faiss HINTS /usr/local/lib)
find_path(FAISS_INCLUDE faiss/IndexFlat.h HINTS /usr/local/include)

# ONNX Runtime (optional)
find_library(ORT_LIB onnxruntime HINTS /usr/local/lib /usr/lib/aarch64-linux-gnu)
find_path(ORT_INCLUDE onnxruntime_cxx_api.h HINTS /usr/local/include/onnxruntime)

# ──────────────────────────────────────────────
# NETRA v3 executable
# ──────────────────────────────────────────────
set(NETRA_SOURCES
    src/main.cpp
    src/llm_engine.cpp
    src/rag_engine.cpp
    src/asr_engine.cpp
    src/tts_engine.cpp
    src/ocr_engine.cpp
    src/server.cpp
    src/utils.cpp
)

add_executable(netra ${NETRA_SOURCES})

target_include_directories(netra PRIVATE include)

if(FAISS_INCLUDE)
    target_include_directories(netra PRIVATE ${FAISS_INCLUDE})
endif()

if(ORT_INCLUDE)
    target_include_directories(netra PRIVATE ${ORT_INCLUDE})
endif()

target_link_libraries(netra PRIVATE
    llama
    httplib
    nlohmann_json
    Threads::Threads
)

if(NETRA_BUILD_ASR)
    target_link_libraries(netra PRIVATE whisper)
    target_compile_definitions(netra PRIVATE NETRA_HAS_ASR)
endif()

if(FAISS_LIB)
    target_link_libraries(netra PRIVATE ${FAISS_LIB})
    target_compile_definitions(netra PRIVATE NETRA_HAS_FAISS)
endif()

if(ORT_LIB)
    target_link_libraries(netra PRIVATE ${ORT_LIB})
    target_compile_definitions(netra PRIVATE NETRA_HAS_ORT)
endif()

if(NETRA_USE_CUDA)
    target_compile_definitions(netra PRIVATE NETRA_HAS_CUDA)
endif()

install(TARGETS netra DESTINATION bin)
