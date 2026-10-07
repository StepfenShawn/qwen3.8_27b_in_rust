# qwen3.8_27b_in_rust
Run the native Qwen3.8 27B LLM locally on one laptop CPU: pure rust, no GPU!  

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