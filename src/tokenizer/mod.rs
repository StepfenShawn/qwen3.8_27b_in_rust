use ahash::AHashMap;
use std::fmt::{self};
use tokenizers::{
    AddedToken, Tokenizer, decoders::byte_level::ByteLevel as ByteLevelDecoder,
    models::bpe::BpeBuilder, pre_tokenizers::byte_level::ByteLevel,
};

use crate::gguf::{Gguf, MetaEntry, MetaType};

#[derive(Debug)]
pub enum TokenizerError {
    MissingField(String),
    ParseError(String),
    TokenizerError(String),
    Utf8Error(std::str::Utf8Error),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::MissingField(s) => write!(f, "MissingField: {}", s),
            TokenizerError::ParseError(s) => write!(f, "ParseError: {}", s),
            TokenizerError::TokenizerError(s) => write!(f, "TokenizerError: {}", s),
            TokenizerError::Utf8Error(e) => write!(f, "Utf8Error: {}", e),
        }
    }
}

impl std::error::Error for TokenizerError {}

pub fn build_tokenizer_from_gguf(gguf: &Gguf) -> Result<Tokenizer, TokenizerError> {
    // 1. tokens
    let tokens_entry = gguf.find_meta("tokenizer.ggml.tokens").ok_or_else(|| {
        TokenizerError::MissingField("Missing `tokenizer.ggml.tokens` in .gguf file".into())
    })?;

    let tokens = extract_string_array(tokens_entry)?;

    let merges_entry = gguf.find_meta("tokenizer.ggml.merges").ok_or_else(|| {
        TokenizerError::MissingField("GGUF 中缺少 tokenizer.ggml.merges 字段".into())
    })?;
    let merges_raw = extract_string_array(merges_entry)?;

    let token_types = gguf
        .find_meta("tokenizer.ggml.token_type")
        .and_then(|e| extract_i32_array(e).ok());

    let unk_token = find_special_token(&tokens, &token_types, "unknown")
        .unwrap_or_else(|| "<|endoftext|>".to_string());

    let vocab: AHashMap<String, u32> = tokens
        .iter()
        .enumerate()
        .map(|(i, t)| (t.clone(), i as u32))
        .collect();

    let merges: Vec<(String, String)> = merges_raw
        .iter()
        .map(|m| {
            let mut it = m.splitn(2, ' ');
            let a = it.next().unwrap_or("").to_string();
            let b = it.next().unwrap_or("").to_string();
            (a, b)
        })
        .collect();

    let bpe = BpeBuilder::new()
        .vocab_and_merges(vocab, merges)
        .unk_token(unk_token)
        .build()
        .map_err(|e| TokenizerError::TokenizerError(format!("构建 BPE 模型失败: {e}")))?;

    let mut tokenizer = Tokenizer::new(bpe);

    let pre_tok = ByteLevel::new(false, false, true);
    tokenizer.with_pre_tokenizer(Some(pre_tok));

    let decoder = ByteLevelDecoder::new(false, false, true);
    tokenizer.with_decoder(Some(decoder));

    if let Some(types) = &token_types {
        register_special_tokens(&mut tokenizer, &tokens, types)?;
    }

    Ok(tokenizer)
}

pub fn get_all_control_token_ids(gguf: &Gguf) -> Vec<u32> {
    gguf.find_meta("tokenizer.ggml.token_type")
        .and_then(|entry| extract_i32_array(entry).ok())
        .map(|types| {
            types
                .iter()
                .enumerate()
                .filter(|&(_, &token_type)| token_type == 3)
                .map(|(id, _)| id as u32)
                .collect()
        })
        .unwrap_or_default()
}

/// 从 MetaEntry 中提取字符串数组
fn extract_string_array(entry: &MetaEntry<'_>) -> Result<Vec<String>, TokenizerError> {
    if entry.ty != MetaType::Array {
        return Err(TokenizerError::ParseError(format!(
            "期望 Array 类型，但得到 {:?}",
            entry.ty
        )));
    }
    if entry.array_type != Some(MetaType::String) {
        return Err(TokenizerError::ParseError(format!(
            "期望 String 数组，但得到 {:?}",
            entry.array_type
        )));
    }

    let mut result = Vec::with_capacity(entry.count as usize);
    let mut pos = 0usize;
    let data = entry.data;

    for _ in 0..entry.count {
        if pos + 8 > data.len() {
            return Err(TokenizerError::ParseError(format!("字符串长度前缀越界")));
        }
        let len = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap()) as usize;
        pos += 8;
        if pos + len > data.len() {
            return Err(TokenizerError::ParseError(format!("字符串内容越界")));
        }
        let s =
            std::str::from_utf8(&data[pos..pos + len]).map_err(|e| TokenizerError::Utf8Error(e))?;
        result.push(s.to_string());
        pos += len;
    }

    Ok(result)
}

/// 从 MetaEntry 中提取 i32 数组（用于 token_type）
fn extract_i32_array(entry: &MetaEntry<'_>) -> Result<Vec<i32>, TokenizerError> {
    if entry.ty != MetaType::Array || entry.array_type != Some(MetaType::Int32) {
        return Err(TokenizerError::ParseError("期望 Int32 数组".into()));
    }
    let mut result = Vec::with_capacity(entry.count as usize);
    let mut pos = 0usize;
    for _ in 0..entry.count {
        if pos + 4 > entry.data.len() {
            return Err(TokenizerError::ParseError("i32 数组越界".into()));
        }
        result.push(i32::from_le_bytes(
            entry.data[pos..pos + 4].try_into().unwrap(),
        ));
        pos += 4;
    }
    Ok(result)
}

/// 根据 token_type 查找特殊 token
/// GGUF token_type 枚举: 1=Normal, 2=Unknown, 3=Control, 4=UserDefined, 5=Unused, 6=Byte
fn find_special_token(
    tokens: &[String],
    token_types: &Option<Vec<i32>>,
    kind: &str,
) -> Option<String> {
    let types = token_types.as_ref()?;
    let target_type = match kind {
        "unknown" => 2,
        "control" => 3,
        _ => return None,
    };
    for (i, &t) in types.iter().enumerate() {
        if t == target_type {
            return Some(tokens[i].clone());
        }
    }
    None
}

fn register_special_tokens(
    tokenizer: &mut Tokenizer,
    tokens: &[String],
    token_types: &[i32],
) -> Result<(), TokenizerError> {
    let added: Vec<AddedToken> = token_types
        .iter()
        .enumerate()
        .filter(|&(_, &t)| t == 3 || t == 4) // Control or UserDefined
        .map(|(i, _)| AddedToken::from(tokens[i].as_str(), true))
        .collect();

    if !added.is_empty() {
        tokenizer
            .add_special_tokens(added)
            .map_err(|e| TokenizerError::TokenizerError(format!("注册特殊 token 失败: {e}")))?;
    }
    Ok(())
}
