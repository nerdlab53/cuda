# cuda

### **Week 1-2: GPU Computing & CUDA Fundamentals**
**Goal**: Understand GPU architecture and write basic CUDA programs.  
**Resources**:  
- **Blogs**:  
  - [NVIDIA’s Introduction to CUDA](https://developer.nvidia.com/cuda-zone)  
  - [Why GPUs? A Beginner’s Guide](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)  
- **Tutorials**:  
  - [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) (Chapters 1-3)  
  - [CUDA Quickstart](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (Hello World, Vector Addition)  
- **Code Exercises**:  
  - Write a "Hello World" CUDA kernel.  
  - Implement vector addition (`a + b = c`) on GPU.  
  - Profile code with `nvprof` to measure runtime.  

---

### **Week 3-4: CUDA Memory Management & Optimization**  
**Goal**: Master memory hierarchies (global, shared, constant) and optimize kernels.  
**Resources**:  
- **Blogs**:  
  - [Understanding CUDA Memory Types](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)  
  - [Shared Memory & Bank Conflicts](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)  
- **Tutorials**:  
  - [CUDA Matrix Multiplication](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory) (with shared memory optimization)  
  - [CUDA Streams and Concurrency](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)  
- **Code Exercises**:  
  - Optimize matrix multiplication using shared memory.  
  - Implement tiled convolution (e.g., image blurring).  
  - Use CUDA streams to overlap data transfers and computation.  

---

### **Week 5-6: Advanced CUDA & Libraries**  
**Goal**: Learn CUDA libraries (cuBLAS, cuFFT) and debug/profiling tools.  
**Resources**:  
- **Blogs**:  
  - [Accelerating Linear Algebra with cuBLAS](https://developer.nvidia.com/cublas)  
  - [Nsight Systems for Profiling](https://developer.nvidia.com/nsight-systems)  
- **Tutorials**:  
  - [CUDA Thrust Library](https://docs.nvidia.com/cuda/thrust/index.html) (for high-level abstractions)  
  - [NVIDIA Nsight Compute](https://developer.nvidia.com/nsight-compute) (kernel profiling)  
- **Code Exercises**:  
  - Use cuBLAS to accelerate matrix multiplication.  
  - Profile a kernel with Nsight Compute and reduce warp divergence.  
  - Implement a reduction kernel (e.g., sum of array).  

---

### **Week 7-8: Triton Fundamentals & Integration**  
**Goal**: Transition to Triton for AI/ML workloads and compare with CUDA.  
**Resources**:  
- **Blogs**:  
  - [OpenAI Triton: A GPU Programming Language](https://openai.com/research/triton)  
  - [Triton vs CUDA: When to Use What](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)  
- **Tutorials**:  
  - [Triton Documentation](https://triton-lang.org/main/index.html) (Start with Vector Add, Matrix Multiply)  
  - [Triton for PyTorch Custom Kernels](https://pytorch.org/tutorials/intermediate/triton_tutorial.html)  
- **Code Exercises**:  
  - Rewrite CUDA vector addition in Triton.  
  - Implement fused Softmax kernel in Triton.  
  - Benchmark Triton vs CUDA for matrix multiplication.  

---

### **Final Project (Week 9-10)**  
**Goal**: Build a real-world application using CUDA and Triton.  
**Examples**:  
- **CUDA**: Optimize a physics simulation (e.g., n-body problem).  
- **Triton**: Implement a custom transformer layer for LLMs.  
- **Hybrid**: Use CUDA for preprocessing and Triton for model inference.  

---

### **Additional Tips**  
1. **Community & Support**:  
   - Join the [NVIDIA Developer Forums](https://forums.developer.nvidia.com/) and [Triton GitHub Discussions](https://github.com/openai/triton/discussions).  
2. **Code Repositories**:  
   - Fork and study [CUDA Samples](https://github.com/NVIDIA/cuda-samples) and [Triton Tutorials](https://github.com/openai/triton/tree/main/python/tutorials).  
3. **Debugging Tools**:  
   - Use `cuda-gdb` for debugging and `nsys` for system-wide profiling.  

---

### **Weekly Time Breakdown**  
- **Reading & Theory**: 1-2 hours (blogs, documentation).  
- **Hands-on Coding**: 2-3 hours (exercises, debugging).  
- **Project Work**: 1-2 hours (final weeks).  
