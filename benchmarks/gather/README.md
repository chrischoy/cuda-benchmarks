# CUDA Matrix Gather Operations Benchmark Suite

This directory contains benchmarks for CUDA matrix gather operations, testing various methods for efficiently gathering rows from matrices.

## Results

Matrix Size     | Gather %   | 1st Place                      | 2nd Place                      | 3rd Place                      |
----------------------------------------------------------------------------------------------------------------------------------
1024x8          | 50.0%      | Float2 vectorized (10.08 GB/s) | CUB device gather (10.03 GB/s) | Float8 vectorized (9.96 GB/s)  |
1024x32         | 50.0%      | Float4 vectorized (40.43 GB/s) | Coalesced row (39.63 GB/s)     | Element-wise (39.61 GB/s)      |
1024x64         | 50.0%      | Float4 vectorized (78.65 GB/s) | Float8 vectorized (76.83 GB/s) | Float2 vectorized (76.26 GB/s) |
1024x128        | 50.0%      | Float2 vectorized (146.30 GB/s) | Float8 vectorized (144.77 GB/s) | Float4 vectorized (141.74 GB/s) |
1024x256        | 50.0%      | Coalesced float4 (313.16 GB/s) | Coalesced float8 (293.44 GB/s) | Coalesced column (293.16 GB/s) |
1024x512        | 50.0%      | Float8 vectorized (494.41 GB/s) | Coalesced float4 (494.21 GB/s) | Coalesced float8 (489.06 GB/s) |
16384x8         | 50.0%      | Coalesced float4 (135.63 GB/s) | Coalesced column (134.26 GB/s) | Coalesced row (132.40 GB/s)    |
16384x32        | 50.0%      | Coalesced float8 (521.67 GB/s) | Coalesced float4 (521.22 GB/s) | Shared memory tiled (477.77 GB/s) |
16384x64        | 50.0%      | Coalesced float8 (857.24 GB/s) | Coalesced float4 (839.55 GB/s) | CUB warp gather (801.78 GB/s)  |
16384x128       | 50.0%      | Coalesced float8 (1416.95 GB/s) | Coalesced float4 (1403.91 GB/s) | Float2 vectorized (1370.42 GB/s) |
16384x256       | 50.0%      | Float4 vectorized (2227.05 GB/s) | Coalesced float4 (2164.37 GB/s) | Float2 vectorized (2067.24 GB/s) |
16384x512       | 50.0%      | Float4 vectorized (2376.06 GB/s) | Coalesced float4 (2350.61 GB/s) | Coalesced float8 (1882.17 GB/s) |
262144x8        | 50.0%      | CUB block gather (1211.01 GB/s) | CUB warp gather (1176.01 GB/s) | CUB device gather (1112.00 GB/s) |
262144x32       | 50.0%      | Coalesced float4 (2146.06 GB/s) | Float4 vectorized (1976.65 GB/s) | Float2 vectorized (1960.97 GB/s) |
262144x64       | 50.0%      | Float8 vectorized (1123.29 GB/s) | Coalesced float4 (1119.98 GB/s) | Float4 vectorized (1117.22 GB/s) |
262144x128      | 50.0%      | Float2 vectorized (890.08 GB/s) | Float4 vectorized (887.31 GB/s) | Coalesced float4 (883.88 GB/s) |
262144x256      | 50.0%      | CUB device gather (852.86 GB/s) | Coalesced row (843.94 GB/s)    | Float2 vectorized (836.58 GB/s) |
262144x512      | 50.0%      | Coalesced row (833.75 GB/s)    | CUB device gather (829.72 GB/s) | Float8 vectorized (815.74 GB/s) |
1048576x8       | 50.0%      | Coalesced float4 (1371.38 GB/s) | CUB block gather (1238.90 GB/s) | Coalesced float8 (1225.45 GB/s) |
1048576x32      | 50.0%      | Float4 vectorized (857.48 GB/s) | Float8 vectorized (855.93 GB/s) | Float2 vectorized (853.54 GB/s) |
1048576x64      | 50.0%      | CUB device gather (842.74 GB/s) | Coalesced row (841.63 GB/s)    | Float2 vectorized (824.39 GB/s) |
1048576x128     | 50.0%      | Coalesced row (832.72 GB/s)    | CUB device gather (830.08 GB/s) | Element-wise (817.95 GB/s)     |
1048576x256     | 50.0%      | Coalesced column (821.55 GB/s) | Element-wise (817.38 GB/s)     | Float8 vectorized (805.68 GB/s) |
1048576x512     | 50.0%      | Element-wise (412.60 GB/s)     | Coalesced column (411.39 GB/s) | Float8 vectorized (408.91 GB/s) |

### Benchmark Parameters

The comprehensive benchmark tests:
- **Matrix sizes**: 1024, 16384, 262144, 1048576 rows × 8, 32, 64, 128, 256, 512 columns
- **Gather ratios**: 25%, 50%, 75% of rows
- **Methods**: 12 different gather implementations including element-wise, vectorized, coalesced, shared memory, and CUB variants

## Performance Analysis

### Executive Summary

The gather benchmark evaluates various CUDA gather operations for matrix row extraction, testing 12 different implementations across matrix sizes from 1K to 1M rows. The results show significant performance variations based on matrix dimensions, with **Coalesced float4** and **Float4 vectorized** emerging as the most consistent top performers.

### Key Performance Insights

#### 1. Top Performers by Matrix Size Category

**Small Matrices (1K rows):**
- **Float4 vectorized** dominates for medium-width matrices (32-64 columns)
- **Coalesced float4** excels for wide matrices (256+ columns)
- **Float2 vectorized** shows strong performance for narrow matrices (8 columns)

**Medium Matrices (16K rows):**
- **Coalesced float8** and **Coalesced float4** consistently lead
- **Float4 vectorized** performs exceptionally well for wide matrices (256-512 columns)
- **CUB warp gather** appears in top 3 for 64-column matrices

**Large Matrices (256K rows):**
- **CUB methods** dominate narrow matrices (8 columns)
- **Coalesced float4** maintains strong performance across most sizes
- **Float8 vectorized** shows competitive performance for 64-column matrices

**Very Large Matrices (1M rows):**
- **Coalesced float4** leads for narrow matrices (8 columns)
- **Float4 vectorized** dominates medium-width matrices (32 columns)
- **CUB device gather** and **Coalesced row** methods excel for wider matrices

#### 2. Performance Patterns

**Narrow Matrices (8 columns):**
- **CUB methods** (block gather, warp gather, device gather) dominate large matrices
- **Coalesced float4** performs best for very large matrices
- Performance ranges from 10 GB/s (small) to 1.4 TB/s (very large)

**Medium-Width Matrices (32-128 columns):**
- **Coalesced float4/float8** and **Float4 vectorized** are most consistent
- Performance scales well with matrix size: 40-2.1 TB/s range
- **CUB warp gather** appears competitive for 64-column matrices

**Wide Matrices (256-512 columns):**
- **Float4 vectorized** and **Coalesced float4** lead for medium matrices
- Performance peaks around 2.3 TB/s for 16K×512 matrices
- **Element-wise** and **Coalesced column** methods perform well for very large matrices

#### 3. Method Performance Characteristics

**Vectorized Methods:**
- **Float4 vectorized**: Most consistent top performer across matrix sizes
- **Float8 vectorized**: Strong for medium-width matrices, less effective for very large
- **Float2 vectorized**: Competitive for narrow and medium-width matrices

**Coalesced Methods:**
- **Coalesced float4**: Best overall performer, especially for narrow matrices
- **Coalesced float8**: Excellent for medium matrices, performance drops for very large
- **Coalesced row/column**: Competitive for specific matrix sizes

**CUB Library Methods:**
- **CUB block gather**: Dominates large narrow matrices (256K×8)
- **CUB warp gather**: Strong for medium-width matrices
- **CUB device gather**: Consistent performer across various sizes

**Other Methods:**
- **Element-wise**: Surprisingly competitive for very large wide matrices
- **Shared memory tiled**: Appears in top 3 for 16K×32 matrices

#### 4. Performance Scaling

**Bandwidth Scaling:**
- Small matrices (1K): 10-500 GB/s
- Medium matrices (16K): 135-2.3 TB/s
- Large matrices (256K): 800-2.1 TB/s
- Very large matrices (1M): 400-1.4 TB/s

**Optimal Matrix Sizes:**
- Peak performance occurs at **16K×512** matrices (~2.3 TB/s)
- Performance degrades for very large matrices due to memory bandwidth limitations
- Narrow matrices show good scaling with size due to coalescing benefits

### Optimization Recommendations

#### For Different Matrix Sizes:
- **Small matrices** (< 16K rows): Use **Float4 vectorized** for medium-width, **Coalesced float4** for wide
- **Medium matrices** (16K-256K rows): **Coalesced float4/float8** provide best performance
- **Large matrices** (> 256K rows): **CUB methods** for narrow matrices, **Coalesced float4** for others

#### For Different Matrix Widths:
- **Narrow matrices** (8 columns): **CUB methods** for large matrices, **Coalesced float4** for very large
- **Medium-width matrices** (32-128 columns): **Float4 vectorized** and **Coalesced float4/float8**
- **Wide matrices** (256+ columns): **Float4 vectorized** for medium matrices, **Element-wise** for very large

### Performance Bottlenecks

1. **Memory bandwidth saturation** at ~2.3 TB/s for optimal matrix sizes
2. **Gather pattern overhead** reduces performance for very large matrices
3. **Kernel launch overhead** affects small matrices
4. **Memory access patterns** become critical for narrow matrices

### Conclusion

The gather benchmark reveals that **Coalesced float4** and **Float4 vectorized** methods provide the most consistent high performance across matrix sizes. CUB library methods excel for specific scenarios (large narrow matrices), while vectorized approaches scale well with matrix width. The optimal choice depends heavily on matrix dimensions, with careful consideration needed for the trade-offs between coalescing, vectorization, and library optimizations.

## Analyzing Results

#### Performance vs Matrix Size

```bash
# Generate performance vs N plots for different channel counts
python analyze_results.py results_file.json --plot-performance

# Customize channel counts
python analyze_results.py results_file.json --plot-performance --channels 64 128 256

# Save plots to directory
python analyze_results.py results_file.json --plot-performance --save-plots plots/
```

#### Performance vs Gather Ratio

```bash
# Generate performance vs gather ratio plots
python analyze_results.py results_file.json --plot-gather-ratio

# Specify matrix sizes to plot (N1 C1 N2 C2 format)
python analyze_results.py results_file.json --plot-gather-ratio --sizes 1024 64 4096 128 16384 256

# Save plots
python analyze_results.py results_file.json --plot-gather-ratio --save-plots plots/
```

#### Heatmaps

```bash
# Generate performance heatmap
python analyze_results.py results_file.json --plot-heatmap

# Filter methods by name
python analyze_results.py results_file.json --plot-heatmap --method-filter "CUB"

# Save heatmap
python analyze_results.py results_file.json --plot-heatmap --save-plots plots/
```

### Complete Analysis Example

```bash
# Run comprehensive analysis with all visualizations
python analyze_results.py results_file.json \
    --metadata \
    --stats all \
    --rankings 5 \
    --plot-performance \
    --plot-gather-ratio \
    --plot-heatmap \
    --save-plots ./plots
```
