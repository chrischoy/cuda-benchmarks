#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector_types.h>

/**
 * Custom vectorized data types for efficient CUDA memory operations
 * These types enable vectorized loads/stores for different precisions
 */

// ============================================================================
// Template-based vector types
// ============================================================================

template<typename T, int N>
struct __align__(N * sizeof(T)) vector_type {
    T data[N];

    // Constructors
    __device__ __host__ vector_type() {}

    __device__ __host__ vector_type(T val) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            data[i] = val;
        }
    }

    __device__ __host__ vector_type(const T* ptr) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            data[i] = ptr[i];
        }
    }

    // Add variadic constructor
    template<typename... Args>
    __device__ __host__ vector_type(Args... args) 
        : data{static_cast<T>(args)...} {
        static_assert(sizeof...(args) == N, "Wrong number of arguments");
    }

    // Array access operators
    __device__ __host__ T& operator[](int i) { return data[i]; }
    __device__ __host__ const T& operator[](int i) const { return data[i]; }

    // Arithmetic operators
    __device__ vector_type operator+(const vector_type& other) const {
        vector_type result;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            result.data[i] = data[i] + other.data[i];
        }
        return result;
    }

    __device__ vector_type operator*(const vector_type& other) const {
        vector_type result;
        #pragma unroll
        for (int i = 0; i < N; i++) {
            result.data[i] = data[i] * other.data[i];
        }
        return result;
    }

    __device__ vector_type& operator+=(const vector_type& other) {
        #pragma unroll
        for (int i = 0; i < N; i++) {
            data[i] += other.data[i];
        }
        return *this;
    }
};

// ============================================================================
// Specialized vector types with proper alignment and precision-specific operations
// ============================================================================

// Half8 - 8 half-precision floats (16 bytes, 128-bit aligned)
struct __align__(16) half8 : public vector_type<half, 8> {
    using base = vector_type<half, 8>;
    using base::base;
    
    // Add missing constructor
    __device__ __host__ half8(half a, half b, half c, half d,
                              half e, half f, half g, half h) {
        data[0] = a; data[1] = b; data[2] = c; data[3] = d;
        data[4] = e; data[5] = f; data[6] = g; data[7] = h;
    }

    // Half-precision specific arithmetic operators
    __device__ half8 operator+(const half8& other) const {
        half8 result;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result.data[i] = __hadd(data[i], other.data[i]);
        }
        return result;
    }

    __device__ half8 operator*(const half8& other) const {
        half8 result;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result.data[i] = __hmul(data[i], other.data[i]);
        }
        return result;
    }

    __device__ half8& operator+=(const half8& other) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[i] = __hadd(data[i], other.data[i]);
        }
        return *this;
    }
};

// Half16 - 16 half-precision floats (32 bytes, 256-bit aligned)
struct __align__(32) half16 : public vector_type<half, 16> {
    using base = vector_type<half, 16>;
    using base::base;

    // Half-precision specific arithmetic operators
    __device__ half16 operator+(const half16& other) const {
        half16 result;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = __hadd(data[i], other.data[i]);
        }
        return result;
    }

    __device__ half16 operator*(const half16& other) const {
        half16 result;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = __hmul(data[i], other.data[i]);
        }
        return result;
    }

    __device__ half16& operator+=(const half16& other) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            data[i] = __hadd(data[i], other.data[i]);
        }
        return *this;
    }
};

// Bfloat8 - 8 bfloat16 floats (16 bytes, 128-bit aligned)
struct __align__(16) bfloat8 : public vector_type<__nv_bfloat16, 8> {
    using base = vector_type<__nv_bfloat16, 8>;
    using base::base;

    // Bfloat16-precision specific arithmetic operators
    __device__ bfloat8 operator+(const bfloat8& other) const {
        bfloat8 result;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result.data[i] = __hadd(data[i], other.data[i]);
        }
        return result;
    }

    __device__ bfloat8 operator*(const bfloat8& other) const {
        bfloat8 result;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            result.data[i] = __hmul(data[i], other.data[i]);
        }
        return result;
    }

    __device__ bfloat8& operator+=(const bfloat8& other) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[i] = __hadd(data[i], other.data[i]);
        }
        return *this;
    }
};

// Bfloat16 - 16 bfloat16 floats (32 bytes, 256-bit aligned)
struct __align__(32) bfloat16 : public vector_type<__nv_bfloat16, 16> {
    using base = vector_type<__nv_bfloat16, 16>;
    using base::base;

    // Bfloat16-precision specific arithmetic operators
    __device__ bfloat16 operator+(const bfloat16& other) const {
        bfloat16 result;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = __hadd(data[i], other.data[i]);
        }
        return result;
    }

    __device__ bfloat16 operator*(const bfloat16& other) const {
        bfloat16 result;
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            result.data[i] = __hmul(data[i], other.data[i]);
        }
        return result;
    }

    __device__ bfloat16& operator+=(const bfloat16& other) {
        #pragma unroll
        for (int i = 0; i < 16; i++) {
            data[i] = __hadd(data[i], other.data[i]);
        }
        return *this;
    }
};

// ============================================================================
// Float8 - 8 single-precision floats (32 bytes, 128-bit loads as float4 pairs)
// ============================================================================

struct __align__(16) float8 {
    float4 lo, hi;

    // Constructors
    __device__ __host__ float8() {}

    __device__ __host__ float8(float val) {
        lo = make_float4(val, val, val, val);
        hi = make_float4(val, val, val, val);
    }

    __device__ __host__ float8(float4 l, float4 h) : lo(l), hi(h) {}

    // Array access
    __device__ __host__ float& operator[](int i) {
        return (i < 4) ? (&lo.x)[i] : (&hi.x)[i - 4];
    }

    __device__ __host__ const float& operator[](int i) const {
        return (i < 4) ? (&lo.x)[i] : (&hi.x)[i - 4];
    }

    // Arithmetic operators
    __device__ float8 operator+(const float8& other) const {
        return float8(
            make_float4(lo.x + other.lo.x, lo.y + other.lo.y,
                       lo.z + other.lo.z, lo.w + other.lo.w),
            make_float4(hi.x + other.hi.x, hi.y + other.hi.y,
                       hi.z + other.hi.z, hi.w + other.hi.w)
        );
    }

    __device__ float8 operator*(const float8& other) const {
        return float8(
            make_float4(lo.x * other.lo.x, lo.y * other.lo.y,
                       lo.z * other.lo.z, lo.w * other.lo.w),
            make_float4(hi.x * other.hi.x, hi.y * other.hi.y,
                       hi.z * other.hi.z, hi.w * other.hi.w)
        );
    }
};

// ============================================================================
// Utility Functions
// ============================================================================

// Half8 utilities
__device__ __host__ inline half8 make_half8(half val) {
    return half8(val);
}

__device__ __host__ inline half8 make_half8(half a, half b, half c, half d,
                                           half e, half f, half g, half h) {
    return half8(a, b, c, d, e, f, g, h);
}

// 128-bit vectorized load for half8
__device__ inline half8 load_half8(const half* ptr) {
    half8 result;
    // Use 128-bit vectorized load (4x uint32_t = 16 bytes = 8 halves)
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    *result_uint4 = *ptr_uint4;
    return result;
}

// 128-bit vectorized store for half8
__device__ inline void store_half8(half* ptr, const half8& val) {
    const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
    uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
    *ptr_uint4 = *val_uint4;
}

// Read-only cache load for half8 (ldg)
__device__ inline half8 ldg_half8(const half* ptr) {
    half8 result;
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    *result_uint4 = __ldg(ptr_uint4);
    return result;
}

// Half16 utilities
__device__ __host__ inline half16 make_half16(half val) {
    return half16(val);
}

__device__ __host__ inline half16 make_half16(half a, half b, half c, half d,
                                              half e, half f, half g, half h,
                                              half i, half j, half k, half l,
                                              half m, half n, half o, half p) {
    return half16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

// 256-bit vectorized load for half16
__device__ inline half16 load_half16(const half* ptr) {
    half16 result;
    // Use 256-bit vectorized load (8x uint32_t = 32 bytes = 16 halves)
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    result_uint4[0] = ptr_uint4[0];  // Load first 128-bit block
    result_uint4[1] = ptr_uint4[1];  // Load second 128-bit block
    return result;
}

// 256-bit vectorized store for half16
__device__ inline void store_half16(half* ptr, const half16& val) {
    const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
    uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
    ptr_uint4[0] = val_uint4[0];  // Store first 128-bit block
    ptr_uint4[1] = val_uint4[1];  // Store second 128-bit block
}

// Read-only cache load for half16 (ldg)
__device__ inline half16 ldg_half16(const half* ptr) {
    half16 result;
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    result_uint4[0] = __ldg(&ptr_uint4[0]);  // Load first 128-bit block
    result_uint4[1] = __ldg(&ptr_uint4[1]);  // Load second 128-bit block
    return result;
}

// Bfloat8 utilities
__device__ __host__ inline bfloat8 make_bfloat8(__nv_bfloat16 val) {
    return bfloat8(val);
}

__device__ __host__ inline bfloat8 make_bfloat8(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d,
                                                __nv_bfloat16 e, __nv_bfloat16 f, __nv_bfloat16 g, __nv_bfloat16 h) {
    return bfloat8(a, b, c, d, e, f, g, h);
}

// 128-bit vectorized load for bfloat8
__device__ inline bfloat8 load_bfloat8(const __nv_bfloat16* ptr) {
    bfloat8 result;
    // Use 128-bit vectorized load (4x uint32_t = 16 bytes = 8 bfloat16)
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    *result_uint4 = *ptr_uint4;
    return result;
}

// 128-bit vectorized store for bfloat8
__device__ inline void store_bfloat8(__nv_bfloat16* ptr, const bfloat8& val) {
    const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
    uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
    *ptr_uint4 = *val_uint4;
}

// Read-only cache load for bfloat8 (ldg)
__device__ inline bfloat8 ldg_bfloat8(const __nv_bfloat16* ptr) {
    bfloat8 result;
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    *result_uint4 = __ldg(ptr_uint4);
    return result;
}

// Bfloat16 utilities
__device__ __host__ inline bfloat16 make_bfloat16(__nv_bfloat16 val) {
    return bfloat16(val);
}

__device__ __host__ inline bfloat16 make_bfloat16(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c, __nv_bfloat16 d,
                                                  __nv_bfloat16 e, __nv_bfloat16 f, __nv_bfloat16 g, __nv_bfloat16 h,
                                                  __nv_bfloat16 i, __nv_bfloat16 j, __nv_bfloat16 k, __nv_bfloat16 l,
                                                  __nv_bfloat16 m, __nv_bfloat16 n, __nv_bfloat16 o, __nv_bfloat16 p) {
    return bfloat16(a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p);
}

// 256-bit vectorized load for bfloat16
__device__ inline bfloat16 load_bfloat16(const __nv_bfloat16* ptr) {
    bfloat16 result;
    // Use 256-bit vectorized load (8x uint32_t = 32 bytes = 16 bfloat16)
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    result_uint4[0] = ptr_uint4[0];  // Load first 128-bit block
    result_uint4[1] = ptr_uint4[1];  // Load second 128-bit block
    return result;
}

// 256-bit vectorized store for bfloat16
__device__ inline void store_bfloat16(__nv_bfloat16* ptr, const bfloat16& val) {
    const uint4* val_uint4 = reinterpret_cast<const uint4*>(&val);
    uint4* ptr_uint4 = reinterpret_cast<uint4*>(ptr);
    ptr_uint4[0] = val_uint4[0];  // Store first 128-bit block
    ptr_uint4[1] = val_uint4[1];  // Store second 128-bit block
}

// Read-only cache load for bfloat16 (ldg)
__device__ inline bfloat16 ldg_bfloat16(const __nv_bfloat16* ptr) {
    bfloat16 result;
    const uint4* ptr_uint4 = reinterpret_cast<const uint4*>(ptr);
    uint4* result_uint4 = reinterpret_cast<uint4*>(&result);
    result_uint4[0] = __ldg(&ptr_uint4[0]);  // Load first 128-bit block
    result_uint4[1] = __ldg(&ptr_uint4[1]);  // Load second 128-bit block
    return result;
}

// Float8 utilities
__device__ __host__ inline float8 make_float8(float val) {
    return float8(val);
}

__device__ inline float8 load_float8(const float* ptr) {
    return float8(
        *reinterpret_cast<const float4*>(ptr),
        *reinterpret_cast<const float4*>(ptr + 4)
    );
}

__device__ inline void store_float8(float* ptr, const float8& val) {
    *reinterpret_cast<float4*>(ptr) = val.lo;
    *reinterpret_cast<float4*>(ptr + 4) = val.hi;
}

__device__ inline float8 ldg_float8(const float* ptr) {
    return float8(
        __ldg(reinterpret_cast<const float4*>(ptr)),
        __ldg(reinterpret_cast<const float4*>(ptr + 4))
    );
}

// ============================================================================
// Type traits and size information
// ============================================================================

template<typename T> struct vector_info;

template<> struct vector_info<half8> {
    static constexpr int elements = 8;
    static constexpr int size_bytes = 16;
    using element_type = half;
};

template<> struct vector_info<half16> {
    static constexpr int elements = 16;
    static constexpr int size_bytes = 32;
    using element_type = half;
};

template<> struct vector_info<bfloat8> {
    static constexpr int elements = 8;
    static constexpr int size_bytes = 16;
    using element_type = __nv_bfloat16;
};

template<> struct vector_info<bfloat16> {
    static constexpr int elements = 16;
    static constexpr int size_bytes = 32;
    using element_type = __nv_bfloat16;
};

template<> struct vector_info<float8> {
    static constexpr int elements = 8;
    static constexpr int size_bytes = 32;
    using element_type = float;
};

template<> struct vector_info<float4> {
    static constexpr int elements = 4;
    static constexpr int size_bytes = 16;
    using element_type = float;
};

template<> struct vector_info<float2> {
    static constexpr int elements = 2;
    static constexpr int size_bytes = 8;
    using element_type = float;
};
