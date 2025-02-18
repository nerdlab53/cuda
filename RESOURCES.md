### **Month 1: GPU Programming & Quantization**
#### **Triton**
1. **[Official Triton Tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html)**  
   - Start with 01-vector-add and 02-fused-softmax  
   - Advanced: 05-layer-norm and 06-matrix-multiplication  

2. **[Triton Practice Repo](https://github.com/ShigekiKarita/triton-practice)**  
   - Hands-on kernels for attention and quantization  

3. **[OpenAI Triton Paper](https://arxiv.org/abs/2110.03793)**  
   - Understand design philosophy and compiler tricks  

#### **Quantization**
1. **[NF4 Paper](https://arxiv.org/abs/2305.14314)** (QLoRA)  
   - Focus on Sections 3-4 for data type theory  

2. **[Bitsandbytes Code Walkthrough](https://github.com/TimDettmers/bitsandbytes)**  
   - Study `functional.py` for 8-bit optimizers  

3. **[PyTorch Quantization Docs](https://pytorch.org/docs/stable/quantization.html)**  
   - Master `torch.ao.quantization` for custom workflows

4. **[Tim's Videos On Quantization](https://www.youtube.com/watch?v=2ETNONas068)**

#### **CUDA**
1. **[NVIDIA CUDA C++ Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)**  
   - Chapters 3 (Programming Model) & 5 (Memory Hierarchy)
   - Easier introduction to CUDA [here](https://developer.nvidia.com/blog/even-easier-introduction-cuda/) (Recommended if don't know the architecture of GPUs, addressing)
   - [Unified Memory Concept](https://developer.nvidia.com/blog/unified-memory-in-cuda-6/)
   - [CUDA Refresher : Origins of GPU Programming](https://developer.nvidia.com/blog/cuda-refresher-reviewing-the-origins-of-gpu-computing/)

2. **[CUDA Reductions Master Class](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)**  
   - Optimized patterns for parallel reductions  

3. **[Cutlass Template Library](https://github.com/NVIDIA/cutlass)**  
   - Study `examples/` for GEMM optimizations
  
4. **[Elliot's CUDA Tutorial](https://www.youtube.com/watch?v=86FAWCzIe_4&t=4250s)**

---

### **Month 2: Distributed Training & Compilation**
#### **FSDP2**
1. **[PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)**  
   - Official implementation with Multi-GPU examples  

2. **[Hugging Face FSDP2 PR](https://github.com/huggingface/accelerate/pull/3394)**  
   - Critical code changes: `accelerate/utils/fsdp.py`  

3. **[Microsoft ZeRO Paper](https://arxiv.org/abs/1910.02054)**  
   - Foundational theory behind parameter sharding  

#### **Torch.Compile**
1. **[PyTorch Compilation Deep Dive](https://pytorch.org/get-started/pytorch-2.0/)**  
   - Focus on graph breaks and dynamic shapes  

2. **[TorchInductor Customization](https://pytorch.org/docs/stable/compile/inductor.html)**  
   - Adding new kernel patterns to the compiler  

3. **[Compilation Debugging Blog](https://dev-discuss.pytorch.org/t/torchdynamo-frequently-asked-questions/183)**  
   - Real-world troubleshooting examples  

#### **Performance Optimization**
1. **[Nsight Systems Guide](https://developer.nvidia.com/nsight-systems)**  
   - Profile CUDA streams and memory bottlenecks  

2. **[PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html)**  
   - Pinning memory, async transfers, and `memory_stats()`  

3. **[Activation Checkpointing Paper](https://arxiv.org/abs/1604.06174)**  
   - Theory behind selective recomputation  

---

### **Month 3: Production Systems**
#### **Unsloth Internals**
1. **[Unsloth Architecture Docs](https://docs.unsloth.ai/architecture/)**  
   - Critical components: `LoRA_MLP`, `Fast_RMS_Layernorm`  

2. **[Unsloth GitHub Issues](https://github.com/unslothai/unsloth/issues)**  
   - Filter by "good first issue" and "Windows" tags  

3. **[Hugging Face Trainer Codebase](https://github.com/huggingface/transformers/tree/main/src/transformers/trainer)**  
   - Study how Unsloth extends this (e.g., `trainer_seq2seq.py`)  

#### **Memory Optimization**
1. **[Apple’s Cut Loss Paper](https://machinelearning.apple.com/research/cut-loss)**  
   - Key Algorithm 1 (page 5) for sparse gradients  

2. **[PyTorch Async Engines](https://pytorch.org/docs/stable/generated/torch.cuda.Stream.html)**  
   - Overlapping compute and data transfers  

3. **[Kaggle GPU Survival Guide](https://www.kaggle.com/docs/tpu)**  
   - Cost optimization for T4/P100 instances  

#### **Integration**
1. **[ML Engineering Patterns](https://eugeneyan.com/writing/ml-design-patterns/)**  
   - Batch processing, caching, and pipeline orchestration  

2. **[PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)**  
   - Identify bottlenecks in end-to-end training  

3. **[Real-World LLM Training](https://github.com/microsoft/DeepSpeedExamples/tree/master/training/huggingface/llama)**  
   - Production patterns from DeepSpeed examples  

---

### **Bonus: Community Resources**
- **Discord Channels**:  
  - [PyTorch Developer](https://discuss.pytorch.org/) (Forum)  
  - [MLOps.community](https://mlops.community/) (Networking)  

- **Newsletters**:  
  - [The Batch](https://www.deeplearning.ai/the-batch/) (AI updates)  
  - [Unsloth Blog](https://docs.unsloth.ai/blog/) (Company-specific tips)  

---

### **Learning Strategy**
1. **Code-First Approach**: Always implement papers line-by-line  
2. **Profile Relentlessly**: Use `py-spy` and Nsight on every kernel  
3. **Contribute Early**: Engage with Unsloth maintainers in GitHub discussions  
