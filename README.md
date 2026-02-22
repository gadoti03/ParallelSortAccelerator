# ParallelSortAccelerator

## Overview

**ParallelSortAccelerator** is a C++ project implementing parallel sorting algorithms with multiple acceleration backends:

- Serial CPU backend  
- SIMD CPU backend (SSE4.2 / AVX2 / hybrid 128-256 bit versions)  
- Optional CUDA GPU backend  

The project is configured using CMake and supports compile-time backend selection.

---

## Requirements

- C++17 compatible compiler  
- CMake >= 3.20  
- Optional CUDA Toolkit (if CUDA backend is used)

---

## Build and Run

You have already prepared the Make/CMake configuration.

### 1. Create build directory

```bash
mkdir build
cd build
```

---

### 2. Configure project

Basic configuration (serial backend enabled):

```bash
cmake ..
```

---

### 3. Enable optional backends

#### Enable SIMD backend

Use:

```bash
cmake -DUSE_SIMD=ON ..
```

Then select at least one SIMD implementation:

- SSE4.2 version  
- AVX2 version  
- Hybrid 128-bit + 256-bit version  

Example:

```bash
cmake -DUSE_SIMD=ON -DUSE_SIMD_V2=ON ..
```

Available SIMD flags:

- `USE_SIMD_V1` → SSE4.2 backend  
- `USE_SIMD_V2` → AVX2 backend  
- `USE_SIMD_V3` → Hybrid SIMD backend  

Important: If SIMD is enabled, at least one SIMD version must be selected.

---

#### Enable CUDA backend

If CUDA is installed:

```bash
cmake -DUSE_CUDA_BACKEND=ON ..
```

The configuration process will automatically check CUDA availability.

---

### 4. Compile project

After configuration:

```bash
make -j$(nproc)
```

---

### 5. Run tests

```bash
ctest
```

or

```bash
./tests/<test_executable>
```

---

## Backend Selection Summary

| Backend | Flag |
|---|---|
| Serial CPU | Enabled by default |
| SIMD CPU | `-DUSE_SIMD=ON` |
| CUDA GPU | `-DUSE_CUDA_BACKEND=ON` |

---

## Example Configurations

### Serial only
```bash
cmake ..
make -j$(nproc)
```

---

### SIMD AVX2 acceleration
```bash
cmake -DUSE_SIMD=ON -DUSE_SIMD_V2=ON ..
make -j$(nproc)
```

---

### CUDA acceleration
```bash
cmake -DUSE_CUDA_BACKEND=ON ..
make -j$(nproc)
```

---

## Notes

- Ensure your compiler supports the selected SIMD instruction set.
- If CUDA backend is enabled but CUDA is not detected, the backend will be disabled automatically.
- The project exports compilation commands for IDE integration.

