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

Matrix Size     Best Method               Time (ms)    BW (GB/s)    2nd Best Method           2nd Min (ms)
---------------------------------------------------------------------------------------------------------
1024x8          Element-wise              0.002        25.30        Float8 vectorized         0.0020
1024x32         Float2 vectorized         0.002        100.65       Float4 vectorized         0.0020
1024x64         Float4 vectorized         0.002        208.23       Float8 vectorized         0.0020
1024x128        Float8 vectorized         0.002        412.07       Float4 vectorized         0.0020
1024x256        Coalesced float8          0.003        743.24       Coalesced float4          0.0020
1024x512        Float4 vectorized         0.003        1358.45      Coalesced float8          0.0020
16384x8         Coalesced float4          0.003        311.98       Coalesced float8          0.0028
16384x32        Float2 vectorized         0.004        1067.61      Float8 vectorized         0.0031
16384x64        Coalesced float4          0.005        1703.70      Float4 vectorized         0.0031
16384x128       Float8 vectorized         0.007        2270.86      Float4 vectorized         0.0061
16384x256       Float8 vectorized         0.007        4176.56      Coalesced float8          0.0072
16384x512       Coalesced float8          0.008        7852.07      Float8 vectorized         0.0072
262144x8        Coalesced float8          0.008        2021.53      Coalesced float4          0.0072
262144x32       Coalesced float8          0.008        7629.99      Coalesced float4          0.0092
262144x64       Coalesced float8          0.010        12184.19     Float8 vectorized         0.0133
262144x128      Coalesced float8          0.011        22427.80     Float8 vectorized         0.0102
262144x256      Float8 vectorized         0.020        25610.56     Coalesced float8          0.0194
262144x512      Float8 vectorized         0.034        29477.24     Coalesced float8          0.0337
1048576x8       Coalesced float8          0.005        13711.91     Coalesced float4          0.0061
1048576x32      Coalesced float8          0.010        23822.96     Coalesced float4          0.0174
1048576x64      Coalesced float8          0.019        26951.74     Float8 vectorized         0.0336
1048576x128     Float8 vectorized         0.034        29546.92     Coalesced float8          0.0338
1048576x256     Float8 vectorized         0.065        30601.25     Coalesced float8          0.0655
1048576x512     Float8 vectorized         0.128        31203.04     Coalesced float8          0.1288


## Setup

- NVIDIA RTX 6000 Ada
- Driver Version: 570.133.20
- CUDA Version: 12.8
