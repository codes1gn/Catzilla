## TODOLIST
* unify all implmenetation variants, to make them more readable under one style
* refactor all kernels into utils that can be extract to simplify the implementation
* optimise with all tricks: tiling, caching, coalescing memory access, swizzle, multi-buffer, pipeline, vectorise, tensorise(tensor core)
* beat cublas
* add cutlass, cudnn, tensorrt and beat them
* try micro kernels modularised design
* compare with candle kernels
* optimise to fit Ragdoll
* add llm.nvim
* candidates: tessellation, kaleidoscope, mortise, fractal


## to support choreo implementation
* basic routine for choreo
* naive cuda kernel to align SIMD behaviours with tileflow perspective
* optimise with this project to SOTA level perf

