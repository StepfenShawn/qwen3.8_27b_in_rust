use deepseek_v4_flash_in_rust::safetensors::Safetensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::path::Path::new("E:/llm_models/");
    let st = Safetensor::open(path)?;

    for t in &st.tensors {
        println!(
            "{}: dtype={:?}, shape={:?}, bytes={}",
            t.name,
            t.dtype,
            t.shape(),
            t.byte_size()
        );
        let raw = t.data();
        println!("{:?}", &raw[..raw.len().min(16)]);
    }
    Ok(())
}
