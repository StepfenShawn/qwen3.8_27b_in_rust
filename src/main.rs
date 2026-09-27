mod gguf;
mod tokenizer;

use gguf::Gguf;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let start = Instant::now();
    let path = std::env::args().nth(1).expect("usage: prog model.gguf");
    let gguf = Gguf::open(&path)?;
    println!("{}", gguf.metadata().len());

    let tok = tokenizer::build_tokenizer_from_gguf(&gguf).unwrap();
   
    let text = "Hello World!";
    let enc = tok.encode(text, false).unwrap();
    let ids = enc.get_ids();
    println!("{:?} {}", ids, ids.len());
    let elapsed = start.elapsed();
    println!("耗时: {:?}", elapsed);
    println!("耗时: {} 秒", elapsed.as_secs_f64());
    assert_eq!(tok.decode(enc.get_ids(), true).unwrap(), text);
    Ok(())
}
