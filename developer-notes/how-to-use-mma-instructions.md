
# Developer Notes: Using GPU MMA Instructions

## Overview

This document provides a detailed guide on using GPU Matrix Multiplication and Accumulation (MMA) instructions like `mma.sync`, `ldmatrix`, `stmatrix`, and `movmatrix`. These instructions are designed for high-performance matrix operations on NVIDIA GPUs with Tensor Core support. We will cover the data type conversions, pointer calculations using `threadIdx` and `laneId`, and provide code examples to demonstrate their use.

## 1. MMA Instructions Overview

### 1.1 `mma.sync`

The `mma.sync` instruction performs matrix multiplication and accumulation on matrix fragments stored in registers. It supports various data types and matrix sizes. The instruction format is as follows:

php

复制代码

`mma.sync.aligned.m<M>n<N>k<K>.<layout>.<out_type>.<in_type_A>.<in_type_B>.<out_type> {d}, {a}, {b}, {c};` 

-   `<M>`, `<N>`, `<K>`: Dimensions of the matrix operation.
-   `<layout>`: Specifies the layout of the result matrix, either `row` or `col`.
-   `<out_type>`: Data type of the output matrix (e.g., `f32` for `float`).
-   `<in_type_A>`: Data type of matrix A (e.g., `f16` for `half`).
-   `<in_type_B>`: Data type of matrix B (e.g., `f16` for `half`).
-   `{d}`: Output accumulator fragment.
-   `{a}`, `{b}`: Input matrix fragments.
-   `{c}`: Accumulation matrix.

### 1.2 `ldmatrix`

The `ldmatrix` instruction loads matrix fragments from shared memory into registers. It supports loading different sizes and layouts. The instruction format is:

php

复制代码

`ldmatrix.sync.aligned.m<M>n<N>.<x>.trans.<element_size> {d}, [addr];` 

-   `<M>`, `<N>`: Dimensions of the matrix fragment.
-   `<x>`: Number of matrix fragments to load.
-   `<element_size>`: Size of the matrix elements (e.g., `b16` for `16-bit`).
-   `{d}`: Register destination.
-   `[addr]`: Memory address to load from.

### 1.3 `stmatrix`

The `stmatrix` instruction stores matrix fragments from registers back to shared memory. The format is:

php

复制代码

`stmatrix.sync.aligned.m<M>n<N>.<x>.trans.<element_size> [addr], {s};` 

-   `[addr]`: Memory address to store the data.
-   `{s}`: Source register containing matrix fragment.

### 1.4 `movmatrix`

The `movmatrix` instruction moves matrix fragments between registers. It is used for register-level manipulation of matrix data.

css

复制代码

`movmatrix {dst}, {src};` 

-   `{dst}`: Destination register.
-   `{src}`: Source register.

## 2. Data Type Conversion

Matrix operations using Tensor Cores often involve mixed-precision computations. For example, the input matrices A and B might use `half` precision (`f16`), while the accumulation matrix uses `float` precision (`f32`). Proper data type conversions are crucial for ensuring correct computations.

### 2.1 Converting `float` to `half`

To convert `float` to `half`, use CUDA’s built-in functions:

cpp

复制代码

`half value = __float2half(1.0f);` 

### 2.2 Converting `half` to `unsigned` for Register Storage

MMA instructions often require loading matrix fragments into unsigned registers. You can pack two `half` values into a single `unsigned` value:

cpp

复制代码

`unsigned short half_value = __half_as_short(value); // Convert half to short
unsigned packed_value = half_value | (half_value << 16); // Pack two half values into unsigned` 

## 3. Pointer Calculations Using `threadIdx` and `laneId`

Calculating the correct memory address for loading and storing matrix fragments is essential. This often involves determining the lane ID and the thread index within a warp.

### 3.1 Calculating Lane ID

Each warp has 32 threads, and each thread has a unique lane ID within the warp. The lane ID can be calculated as:

cpp

复制代码

`int laneId = threadIdx.x % 32;`# Developer Notes: Using GPU MMA Instructions

## Overview

This document provides a detailed guide on using GPU Matrix Multiplication and Accumulation (MMA) instructions like `mma.sync`, `ldmatrix`, `stmatrix`, and `movmatrix`. These instructions are designed for high-performance matrix operations on NVIDIA GPUs with Tensor Core support. We will cover the data type conversions, pointer calculations using `threadIdx` and `laneId`, and provide code examples to demonstrate their use.

## 1. MMA Instructions Overview

### 1.1 `mma.sync`

The `mma.sync` instruction performs matrix multiplication and accumulation on matrix fragments stored in registers. It supports various data types and matrix sizes. The instruction format is as follows:

php

复制代码

`mma.sync.aligned.m<M>n<N>k<K>.<layout>.<out_type>.<in_type_A>.<in_type_B>.<out_type> {d}, {a}, {b}, {c};` 

-   `<M>`, `<N>`, `<K>`: Dimensions of the matrix operation.
-   `<layout>`: Specifies the layout of the result matrix, either `row` or `col`.
-   `<out_type>`: Data type of the output matrix (e.g., `f32` for `float`).
-   `<in_type_A>`: Data type of matrix A (e.g., `f16` for `half`).
-   `<in_type_B>`: Data type of matrix B (e.g., `f16` for `half`).
-   `{d}`: Output accumulator fragment.
-   `{a}`, `{b}`: Input matrix fragments.
-   `{c}`: Accumulation matrix.

### 1.2 `ldmatrix`

The `ldmatrix` instruction loads matrix fragments from shared memory into registers. It supports loading different sizes and layouts. The instruction format is:

php

复制代码

`ldmatrix.sync.aligned.m<M>n<N>.<x>.trans.<element_size> {d}, [addr];` 

-   `<M>`, `<N>`: Dimensions of the matrix fragment.
-   `<x>`: Number of matrix fragments to load.
-   `<element_size>`: Size of the matrix elements (e.g., `b16` for `16-bit`).
-   `{d}`: Register destination.
-   `[addr]`: Memory address to load from.

### 1.3 `stmatrix`

The `stmatrix` instruction stores matrix fragments from registers back to shared memory. The format is:

php

复制代码

`stmatrix.sync.aligned.m<M>n<N>.<x>.trans.<element_size> [addr], {s};` 

-   `[addr]`: Memory address to store the data.
-   `{s}`: Source register containing matrix fragment.

### 1.4 `movmatrix`

The `movmatrix` instruction moves matrix fragments between registers. It is used for register-level manipulation of matrix data.

css

复制代码

`movmatrix {dst}, {src};` 

-   `{dst}`: Destination register.
-   `{src}`: Source register.

## 2. Data Type Conversion

Matrix operations using Tensor Cores often involve mixed-precision computations. For example, the input matrices A and B might use `half` precision (`f16`), while the accumulation matrix uses `float` precision (`f32`). Proper data type conversions are crucial for ensuring correct computations.

### 2.1 Converting `float` to `half`

To convert `float` to `half`, use CUDA’s built-in functions:

cpp

复制代码

`half value = __float2half(1.0f);` 

### 2.2 Converting `half` to `unsigned` for Register Storage

MMA instructions often require loading matrix fragments into unsigned registers. You can pack two `half` values into a single `unsigned` value:

cpp

复制代码

`unsigned short half_value = __half_as_short(value); // Convert half to short
unsigned packed_value = half_value | (half_value << 16); // Pack two half values into unsigned` 

## 3. Pointer Calculations Using `threadIdx` and `laneId`

Calculating the correct memory address for loading and storing matrix fragments is essential. This often involves determining the lane ID and the thread index within a warp.

### 3.1 Calculating Lane ID

Each warp has 32 threads, and each thread has a unique lane ID within the warp. The lane ID can be calculated as:

cpp

复制代码

`int laneId = threadIdx.x % 32;`

## Example：use f32 = f16*f16 + f32 mma.sync instruction to perform matmul

Let us simplify the strategy, we use Mtile x Ntile x Ktile as the tile shape at Shared memory. we thus will tile the data blocks from input global memory to the corresponding SM buffer. This is a minimal case, so we only use one tensorcore operation, which involves 32 consecutive threads in a wrap. Code are given below:
```cpp
const int M_TILE = 16;
const int K_TILE = 8;
const int N_TILE = 8;
const int X_THREAD = 32;
...
// code to launch kernel
```
We use "catz" DSL to perform the simplified tiling strategy on data tileflow in DMA style. Code is
``` cpp
// define shape constants
auto lhs_shape = make_coord(M, K);
auto rhs_shape = make_coord(K, N);
auto out_shape = make_coord(M, N);

auto lhs_sm_tile_shape = Coord(M_TILE, K_TILE);
auto rhs_sm_tile_shape = Coord(K_TILE, N_TILE);
auto out_sm_tile_shape = Coord(M_TILE, N_TILE);

// wraps global memory as Matrix
Matrix<float> lhs_mat = Matrix<float>(lhs, lhs_shape);
Matrix<float> rhs_mat = Matrix<float>(rhs, rhs_shape);
Matrix<float> out_mat = Matrix<float>(out, out_shape);

// use MACRO to define buffer at SM, and wrap into Matrix simutaneously
MAKE_SHARED(lhs_shared_mat, M_TILE, K_TILE, half);
MAKE_SHARED(rhs_shared_mat, K_TILE, N_TILE, half);
MAKE_SHARED(out_shared_mat, M_TILE, N_TILE, float);

// tile K dimension
for (int ko = 0; ko < CEIL_DIV(K, K_TILE); ko++) {
    // we have 32 threads, thus, we need to tile 32 consecutive elements to pass down
    // you can also use 128 elements for vectorised load 'vload' instruction.
    for (int m = 0; m < CEIL_DIV(M_TILE, 2); m++) {
#pragma unroll
      for (int kin = 0; kin < CEIL_DIV(K_TILE, 16); kin++) {
        // use .tile(tile_var: Coord, tile_shape: Coord) to slice a memview from memory
        // when the tile slice has 32 elements, use .dist_to_thread() to map each thread to a data loc.
        lhs_shared_mat.tile_ex(Coord(m, kin), Coord(2, 16)).dist_to_thread()
          = lhs_mat.tile_ex(Coord(blockIdx.x, ko), lhs_sm_tile_shape)
              .tile_ex(Coord(m, kin), Coord(2, 16))
              .dist_to_thread();
      }
    }
    for (int kin = 0; kin < CEIL_DIV(K_TILE, 4); kin++) {
#pragma unroll
      for (int n = 0; n < CEIL_DIV(N_TILE, 8); n++) {
        rhs_shared_mat.tile_ex(Coord(kin, n), Coord(4, 8)).dist_to_thread()
          = rhs_mat.tile_ex(Coord(ko, blockIdx.y), rhs_sm_tile_shape)
              .tile_ex(Coord(kin, n), Coord(4, 8))
              .dist_to_thread();
      }
    }
    __syncthreads();
    ...
    // launch inner kernel to compute
}
```
SOME WORDS TO BE ADDED

Then we will use the wrapped inner kernel to compute, which is impl'd by mma instructions.
Let us take mma_m16n8k16_f16f32 as example, it takes lhs Matrix with half* data type and 16x16 shape, rhs Matrix with half* data type and 16x8 shape, the output is float* Matrix with 16x8 shape.

In its core, we will use 'mma.m16n8k16' instruction for the computation:
```cpp
  // the typing constraints of such instruction is actually (f16x2, f16x2, f32) -> f32 
  // or (bf16x2, bf16x2, f32) -> f32, so we need to use unsigned or uint32_t as the type,
  // and use 'r' type constraints for the register type
  unsigned A[4];
  unsigned B[2];
  float C[4];
  float D[4];

  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               " { %0, %1, %2, %3 }, "
               " { %4, %5, %6, %7 }, "
               " { %8, %9 }, "
               " { %10, %11, %12, %13 };"
               : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
               : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]), "r"(B[0]),
                 "r"(B[1]), "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));
   ```
   One thing need to notice: the A, B, C, and D are all registers (although virtual register at this stage), the data has not loaded from SM to the register. We have to coding this logic by our own.
   One way is the trivial data transfer operations ( = , vload, ... ) in CUDA api or specific 'ldmatrix' instruction.
   
If we want to use the trivial way, we have to figure out, which matrix fragment (this is the exact term how NVIDIA calls) are loaded to the corresponding threads. For 'm16n8k16' instruction, the C matrix are organised into 8x8 matrices with f32 elements. and each threads takes 2 consecutive float register with 64 bits width. Since in our case, the C matrix has 16x8 shape, it extends 8x8 matrix along with 'M' dimension. Here we need to store two f32x2 register to the first 'position' of each 8x8 matrix (position refers to float* in our example).

We thus have store data code in:
```cpp
  d[row_c * 8 + col_c] = D[0];
  d[row_c * 8 + col_c + 1] = D[1];
  d[row_c_ex * 8 + col_c] = D[2];
  d[row_c_ex * 8 + col_c + 1] = D[3];
```
Of course, you can use vectorised store in this case by modify the 'd' matrix with float2 dtype, and packs D[0], D[1] together and D[2], D[3] together.

TODO: add statements
```cpp
  int lane_id = threadIdx.x % 32;
  int lorr = lane_id / 16;
  int lorr_id = lane_id % 16;
  const half* a_new = a + lorr_id * 16 + lorr * 8;
  
  // TODO: pack all these abstractions back to Matrix
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(A[0]), "=r"(A[1]), "=r"(A[2]), "=r"(A[3])
      : "r"(get_smem_ptr(a_new)));

  // NOTE: TO-T16:, point at header loc of each row: matrix has 16x16 of halfs, forms 4 matrices, each
  // has 8x8 halfs, thus the row-stride == 16,
  // if use x2, only T0-T16 involves into datamove, we need T0-T31 together, we need X4
  // T16-T32:, point as mid point of each row.
  //
  // int offset = lane_id * 16;
  // const half *a_lhs = a + offset;
  // const half *a_rhs = a + offset + 8;
  // asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
  //              : "=r"(A[0]), "=r"(A[1])
  //              : "r"(get_smem_ptr(a_lhs)));
  // asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
  //              : "=r"(A[2]), "=r"(A[3])
  //              : "r"(get_smem_ptr(a_rhs)));
```

```cpp
  int uord = lane_id / 8;
  int uord_id = lane_id % 8;
  const half* b_new = b + uord * 8 + uord_id * 16;

  // b matrix is row-major at SM level, but we need it become col-major at wrap level. 
  // we thus need to add .trans qualifier
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
               : "=r"(B[0]), "=r"(B[1])
               : "r"(get_smem_ptr(b_new)));
```
