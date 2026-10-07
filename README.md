# qwen3.8_27b_in_rust
Run the Qwen3.8 27B LLM on one laptop CPU. Written in pure rust: no BLAS, no framework, no GPU!  

# Requirements
| | |
| --- | --- |
| OS  | Linux/x86-64, Windows/x86-64, MacOS/arm is coming soon! |
| CPU | AVX2 + FMA on x86-64 is better, NEON on arm64 is coming soon! |
| GPU | NO |
| RAM | >=8GB |
| Storage | ~20GB free |

# Usage
```
qwen38_27b_in_rust --model MODEL.gguf [--prompt TEXT] [options]
  --prompt TEXT       run one request
  --system TEXT       system instruction
  --max-tokens N      maximum generated tokens (default: 1024)
  --context N         context capacity (default: 4096)
  --temperature N     0 for greedy; default: 1.0
  --top-k N           sample from the best N tokens (default: 20)
  --top-p N           nucleus probability (default: 0.95)
  --presence-penalty N
  --no-thinking       answer directly instead of showing reasoning
  --reasoning-effort N  low, medium, or xhigh (default: xhigh)
  --threads N         worker threads for the kernels (default: one per core,
                      also honoured through RAYON_NUM_THREADS)
  --seed N            sampling seed
```
example:  
```
> ./qwen38_27b_in_rust --model Qwen3.8-27B-UD-Q4_K_M.gguf --prompt Hello --max-tokens 256 --context 256 --no-thinking
Hello! How can I help you today?
[prompt=13 tokens, output=9 tokens, threads=12, elapsed=158.583s
 TTFT=19.100s
 TPOT=15.323s
```

# Build
```
git clone https://github.com/StepfenShawn/qwen3.8_27b_in_rust.git
cd qwen3.8_27b_in_rust
cargo build --release
```

# Fetch the checkpoint
* [Qwen3.8-27B-UD-Q4_K_M](https://www.modelscope.cn/models/unsloth/Qwen3.8-27B-GGUF/resolve/ba7608d4e5e1f3ea3d016cebd1c972c42686e9da/Qwen3.8-27B-UD-Q4_K_M.gguf)

# Benchmark
Coming soon!  