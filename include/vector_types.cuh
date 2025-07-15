#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <vector_types.h>

/**
 * Custom vectorized data types for efficient CUDA memory operations
 * These types enable 128-bit vectorized loads/stores for different precisions
 */

// ============================================================================
// Half8 - 8 half-precision floats (16 bytes, 128-bit aligned)
// ============================================================================

struct __align__(16) half8 {
    half data[8];

    // Constructors
    __device__ __host__ half8() {}

    __device__ __host__ half8(half val) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[i] = val;
        }
    }

    __device__ __host__ half8(half a, half b, half c, half d,
                             half e, half f, half g, half h) {
        data[0] = a; data[1] = b; data[2] = c; data[3] = d;
        data[4] = e; data[5] = f; data[6] = g; data[7] = h;
    }

    __device__ __host__ half8(const half* ptr) {
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            data[i] = ptr[i];
        }
    }

    // Array access operators
    __device__ __host__ half& operator[](int i) { return data[i]; }
    __device__ __host__ const half& operator[](int i) const { return data[i]; }

    // Arithmetic operators
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
