mod gguf;
mod kernel;
mod model;
mod tokenizer;

use gguf::Gguf;
use kernel::Q38Iq1sRepack;
use model::{Q38Model, Q38ModelOps};
use std::io::{self, Write};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::model::sampler::Sampler;

const EFFORT_LOW: i32 = 0;
const EFFORT_MEDIUM: i32 = 1;
const EFFORT_XHIGH: i32 = 2;

struct Options {
    model: Option<String>,
    prompt: Option<String>,
    system: Option<String>,
    context: u32,
    max_tokens: u32,
    seed: u64,
    temperature: f32,
    top_k: u32,
    top_p: f32,
    presence_penalty: f32,
    thinking: bool,
    reasoning_effort: i32,
    threads: Option<usize>,
}

impl Default for Options {
    fn default() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        Options {
            model: None,
            prompt: None,
            system: None,
            context: 4096,
            max_tokens: 1024,
            seed,
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            presence_penalty: 0.0,
            thinking: true,
            reasoning_effort: EFFORT_XHIGH,
            threads: None,
        }
    }
}

fn usage() {
    eprintln!(
        "usage: qwen38 --model MODEL.gguf [--prompt TEXT] [options]\n\
         \x20 --prompt TEXT       run one request\n\
         \x20 --system TEXT       system instruction\n\
         \x20 --max-tokens N      maximum generated tokens (default: 1024)\n\
         \x20 --context N         context capacity (default: 4096)\n\
         \x20 --temperature N     0 for greedy; default: 1.0\n\
         \x20 --top-k N           sample from the best N tokens (default: 20)\n\
         \x20 --top-p N           nucleus probability (default: 0.95)\n\
         \x20 --presence-penalty N\n\
         \x20 --no-thinking       answer directly instead of showing reasoning\n\
         \x20 --reasoning-effort N  low, medium, or xhigh (default: xhigh)\n\
         \x20 --threads N         worker threads for the kernels (default: one per core,\n\
         \x20                     also honoured through RAYON_NUM_THREADS)\n\
         \x20 --seed N            sampling seed"
    );
}

fn parse_args() -> Result<Options, String> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut opts = Options::default();
    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        let mut next = |name: &str| -> Result<String, String> {
            i += 1;
            args.get(i)
                .cloned()
                .ok_or_else(|| format!("{name} requires a value"))
        };
        match arg.as_str() {
            "--model" => opts.model = Some(next("--model")?),
            "--prompt" => opts.prompt = Some(next("--prompt")?),
            "--system" => opts.system = Some(next("--system")?),
            "--max-tokens" => {
                opts.max_tokens = next("--max-tokens")?
                    .parse()
                    .map_err(|_| "invalid --max-tokens".to_string())?
            }
            "--context" => {
                opts.context = next("--context")?
                    .parse()
                    .map_err(|_| "invalid --context".to_string())?
            }
            "--seed" => {
                opts.seed = next("--seed")?
                    .parse()
                    .map_err(|_| "invalid --seed".to_string())?
            }
            "--temperature" => {
                opts.temperature = next("--temperature")?
                    .parse()
                    .map_err(|_| "invalid --temperature".to_string())?
            }
            "--top-k" => {
                opts.top_k = next("--top-k")?
                    .parse()
                    .map_err(|_| "invalid --top-k".to_string())?
            }
            "--top-p" => {
                opts.top_p = next("--top-p")?
                    .parse()
                    .map_err(|_| "invalid --top-p".to_string())?
            }
            "--presence-penalty" => {
                opts.presence_penalty = next("--presence-penalty")?
                    .parse()
                    .map_err(|_| "invalid --presence-penalty".to_string())?
            }
            "--reasoning-effort" => {
                let effort = next("--reasoning-effort")?;
                opts.reasoning_effort = match effort.as_str() {
                    "low" => EFFORT_LOW,
                    "medium" => EFFORT_MEDIUM,
                    "xhigh" => EFFORT_XHIGH,
                    _ => return Err("invalid --reasoning-effort".to_string()),
                };
            }
            "--no-thinking" => opts.thinking = false,
            "--threads" => {
                let threads: usize = next("--threads")?
                    .parse()
                    .map_err(|_| "invalid --threads".to_string())?;
                if threads == 0 {
                    return Err("--threads must be at least 1".to_string());
                }
                opts.threads = Some(threads);
            }
            "--help" | "-h" => {
                usage();
                std::process::exit(0);
            }
            other if !other.starts_with("--") && opts.model.is_none() => {
                opts.model = Some(other.to_string());
            }
            other => return Err(format!("unknown argument: {other}")),
        }
        i += 1;
    }
    if opts.model.is_none() {
        return Err("--model is required".to_string());
    }
    if opts.prompt.is_none() {
        return Err("--prompt is required".to_string());
    }
    Ok(opts)
}

fn render_prompt(user: &str, system: Option<&str>, thinking: bool, effort: i32) -> String {
    const XHIGH: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
    const LOW: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion without unnecessary elaboration.";

    let instruction = if thinking && effort == EFFORT_XHIGH {
        XHIGH
    } else if thinking && effort == EFFORT_LOW {
        LOW
    } else {
        ""
    };
    let system_text = system.unwrap_or("");
    let add_system = !instruction.is_empty() || !system_text.is_empty();

    let mut prompt = String::new();
    if add_system {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(instruction);
        if !instruction.is_empty() && !system_text.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(system_text);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(user);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt.push_str(if thinking {
        "<think>\n"
    } else {
        "<think>\n\n</think>\n\n"
    });
    prompt
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let options = parse_args()?;
    // The kernels draw from rayon's global pool, so `--threads` has to be
    // installed before the first parallel region runs.
    if let Some(threads) = options.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .map_err(|error| format!("unable to start {threads} worker threads: {error}"))?;
    }
    let model_path = options.model.as_deref().unwrap();

    let mut gguf = Gguf::open(model_path)?;
    // IQ1_S checkpoints get a second, SIMD-friendly view of their weight
    // blocks. The mapping stays authoritative; this only adds derived state.
    if let Err(error) = gguf.prepare_iq1_s_repacks() {
        eprintln!("qwen38: IQ1 runtime repack unavailable; using packed weights ({error})");
    }
    let tokenizer = tokenizer::build_tokenizer_from_gguf(&gguf)?;
    let eos = gguf.meta_u32("tokenizer.ggml.eos_token_id").unwrap_or(0);

    let mut model =
        Q38Model::open_gguf(&gguf, options.context).map_err(|_| "unable to open model")?;

    let prompt = render_prompt(
        options.prompt.as_deref().unwrap(),
        options.system.as_deref(),
        options.thinking,
        options.reasoning_effort,
    );
    let encoding = tokenizer.encode(prompt.as_str(), false)?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        return Err("prompt produced no tokens".into());
    }

    let vocab = model.vocab_size() as usize;
    let mut sampler = Sampler::new(
        options.seed,
        options.temperature,
        options.top_k,
        options.top_p,
        options.presence_penalty,
        vocab,
    );
    for &id in &ids {
        sampler.observe(id);
    }

    let started = Instant::now();
    let mut logits = model.prefill(&ids).map_err(|_| "prefill failed")?;
    let mut first_ready: Option<Instant> = None;
    let mut last_ready = started;
    let mut generated = 0u32;
    let mut truncated = false;

    if options.thinking {
        print!("<think>\n");
    }

    while generated < options.max_tokens {
        let token = sampler.sample(logits);
        sampler.observe(token);
        if token == eos {
            break;
        }
        if generated == 0 {
            first_ready = Some(Instant::now());
        }
        let text = tokenizer.decode(&[token], true)?;
        print!("{text}");
        io::stdout().flush().ok();
        generated += 1;
        last_ready = Instant::now();
        if model.position() >= model.context_length() {
            eprintln!(
                "qwen38: context is full ({} positions); restart with a larger --context",
                model.context_length()
            );
            break;
        }
        logits = model.forward_token(token).map_err(|_| "forward failed")?;
    }

    truncated = generated >= options.max_tokens;
    if truncated && options.thinking {
        print!("\n</think>\n\n");
    }
    print!("\n");
    io::stdout().flush().ok();

    let elapsed = started.elapsed().as_secs_f64();
    eprintln!(
        "[prompt={} tokens, output={} tokens, threads={}, elapsed={:.3}s",
        ids.len(),
        generated,
        rayon::current_num_threads(),
        elapsed
    );
    if let Some(first) = first_ready {
        let ttft = first.duration_since(started).as_secs_f64();
        eprintln!(" TTFT={ttft:.3}s");
        if generated > 1 {
            let tpot = last_ready.duration_since(first).as_secs_f64() / (generated - 1) as f64;
            eprintln!(" TPOT={tpot:.3}s");
        }
    }
    if truncated {
        eprintln!(
            "output reached --max-tokens={}; raise the limit to continue longer",
            options.max_tokens
        );
    }
    eprintln!("]");

    Ok(())
}
