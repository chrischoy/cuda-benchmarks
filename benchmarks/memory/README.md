# Matrix Loading Benchmark Results

## Summary

This benchmark compares various CUDA matrix loading strategies across different matrix dimensions, measuring pure loading bandwidth without write-back overhead. All methods store loaded data in shared memory for fair comparison. The results demonstrate the effectiveness of combining vectorized loading with coalesced memory access patterns, with significant performance improvements achieved through proper grid size calculations.

### Key Findings

**Coalesced Float8 Emergence:**
- **Large matrix dominance**: Coalesced float8 consistently leads on larger matrices (262144×8 to 1048576×256)
- **Scaling efficiency**: Achieves 13.7-27.0 GB/s bandwidth on large matrices
- **Grid size fix success**: Now handles the largest matrix (1048576×512) without hanging
- **Optimal for narrow-to-medium widths**: Best choice for matrices with 8-256 columns when row count is large

**Float8 Vectorized Excellence:**
- **Wide matrix champion**: Dominates performance on wide matrices (128+ columns)
- **Peak performance**: Achieves highest bandwidth of 31.2 GB/s on largest matrix (1048576×512)
- **Consistent scaling**: Strong performance across all matrix sizes for wider configurations
- **Memory utilization**: Efficiently utilizes memory bandwidth through 8-element vectorized loads

**Performance Scaling Patterns:**
- **Small matrices (1024×*)**: Element-wise optimal for very small (8 cols), vectorized methods for medium-wide
- **Medium matrices (16384×*)**: Mix of coalesced and vectorized methods depending on width
- **Large matrices (262144×+ rows)**: Clear bifurcation - coalesced float8 for narrow, float8 vectorized for wide
- **Ultra-large matrices (1048576×+ rows)**: Float8 vectorized achieves peak 29-31 GB/s bandwidth

**Method Selection Strategy:**
- **Small matrices (≤1024 rows)**: Use vectorized methods based on width (float2→float4→float8)
- **Medium matrices (16384 rows)**: Coalesced float4/float8 for narrow, vectorized for wide
- **Large matrices (262144+ rows)**: Coalesced float8 for 8-128 columns, float8 vectorized for 256+ columns
- **Critical insight**: Grid size fixes enable proper scaling to largest matrix sizes

**Architecture Insights:**
- **Grid calculation correctness**: Proper element-per-thread accounting crucial for vectorized methods
- **Memory coalescing advantage**: 1D coalesced patterns excel on large datasets
- **Vectorization benefits**: 8-element loads provide substantial bandwidth improvements
- **Scale-dependent optimization**: Different methods optimal at different matrix scales


## Results

| Matrix Size | Best Method | Time (ms) | BW (GB/s) | 2nd Best Method | 2nd Min (ms) |
|-------------|-------------|-----------|-----------|-----------------|--------------|
| 1024x8 | Element-wise | 0.009 | 6.73 | Coalesced column | 0.0082 |
| 1024x32 | Float8 vectorized | 0.009 | 26.32 | Shared memory tiled | 0.0082 |
| 1024x64 | Shared memory tiled | 0.009 | 52.63 | CUB device load | 0.0082 |
| 1024x128 | Shared memory tiled | 0.007 | 131.09 | Coalesced float4 | 0.0065 |
| 1024x256 | Coalesced float4 | 0.008 | 259.95 | Shared memory tiled | 0.0072 |
| 1024x512 | Coalesced float4 | 0.008 | 520.20 | Coalesced row | 0.0072 |
| 16384x8 | Coalesced float4 | 0.007 | 132.00 | Coalesced column | 0.0072 |
| 16384x32 | Coalesced float4 | 0.007 | 523.95 | Coalesced row | 0.0071 |
| 16384x64 | Coalesced float8 | 0.007 | 1045.21 | Coalesced float4 | 0.0071 |
| 16384x128 | Coalesced float8 | 0.007 | 2089.17 | Float8 vectorized | 0.0063 |
| 16384x256 | Float8 vectorized | 0.008 | 4159.83 | Coalesced float8 | 0.0072 |
| 16384x512 | Coalesced float8 | 0.007 | 8365.99 | Float8 vectorized | 0.0072 |
| 262144x8 | Coalesced float8 | 0.004 | 3698.54 | Coalesced float4 | 0.0041 |
| 262144x32 | Coalesced float8 | 0.005 | 12615.46 | Coalesced float4 | 0.0068 |
| 262144x64 | Coalesced float8 | 0.007 | 18061.08 | Float4 vectorized | 0.0102 |
| 262144x128 | Float8 vectorized | 0.011 | 22609.54 | Coalesced float8 | 0.0102 |
| 262144x256 | Float8 vectorized | 0.020 | 25577.86 | Coalesced float8 | 0.0193 |
| 262144x512 | Float8 vectorized | 0.034 | 29350.99 | Coalesced float8 | 0.0338 |
| 1048576x8 | Coalesced float8 | 0.004 | 13990.87 | Coalesced float4 | 0.0061 |
| 1048576x32 | Coalesced float8 | 0.011 | 23668.50 | Coalesced float4 | 0.0178 |
| 1048576x64 | Coalesced float8 | 0.018 | 27254.49 | Float4 vectorized | 0.0335 |
| 1048576x128 | Float8 vectorized | 0.034 | 29481.69 | Coalesced float8 | 0.0338 |
| 1048576x256 | Float8 vectorized | 0.065 | 30597.66 | Coalesced float8 | 0.0655 |
| 1048576x512 | Float8 vectorized | 0.128 | 31191.67 | Coalesced float8 | 0.1299 |

## Setup

- NVIDIA RTX 6000 Ada
- Driver Version: 570.133.20
- CUDA Version: 12.8
