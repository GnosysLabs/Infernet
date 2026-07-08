use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{self, BufReader, Read},
    path::Path,
};

use anyhow::{Context, Result, anyhow, bail};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone)]
pub struct GgufInfo {
    pub version: u32,
    pub metadata_kv_count: u64,
    pub tensor_count: u64,
    pub architecture: Option<String>,
    pub layer_count: Option<u32>,
    pub hidden_size: Option<usize>,
    pub tokenizer_family: Option<String>,
    pub tokenizer_checksum: String,
    pub quantization: Option<String>,
    pub tensor_names: Vec<String>,
}

#[derive(Debug, Clone)]
enum GgufValue {
    UInt(u64),
    Int(i64),
    Float(String),
    Bool(bool),
    String(String),
    Array {
        element_type: u32,
        len: u64,
        checksum_sha256: String,
    },
}

impl fmt::Display for GgufValue {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UInt(value) => write!(formatter, "{value}"),
            Self::Int(value) => write!(formatter, "{value}"),
            Self::Float(value) => write!(formatter, "{value}"),
            Self::Bool(value) => write!(formatter, "{value}"),
            Self::String(value) => formatter.write_str(value),
            Self::Array {
                element_type,
                len,
                checksum_sha256,
            } => write!(
                formatter,
                "array(type={element_type},len={len},sha256={checksum_sha256})"
            ),
        }
    }
}

impl GgufValue {
    fn as_string(&self) -> Option<String> {
        match self {
            Self::String(value) => Some(value.clone()),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::UInt(value) => Some(*value),
            Self::Int(value) if *value >= 0 => Some(*value as u64),
            _ => None,
        }
    }
}

pub fn parse_gguf_info(path: &Path) -> Result<GgufInfo> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);

    let mut magic = [0; 4];
    reader.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }

    let version = read_u32(&mut reader)?;
    let tensor_count = read_u64(&mut reader)?;
    let metadata_kv_count = read_u64(&mut reader)?;
    let mut metadata = BTreeMap::new();

    for _ in 0..metadata_kv_count {
        let key = read_string(&mut reader)?;
        let value_type = read_u32(&mut reader)?;
        let value = read_value(&mut reader, value_type)
            .with_context(|| format!("failed to read GGUF metadata value {key}"))?;
        metadata.insert(key, value);
    }

    let mut tensor_names = Vec::with_capacity(tensor_count.min(usize::MAX as u64) as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut reader)?;
        let dimensions = read_u32(&mut reader)?;
        for _ in 0..dimensions {
            let _ = read_u64(&mut reader)?;
        }
        let _tensor_type = read_u32(&mut reader)?;
        let _offset = read_u64(&mut reader)?;
        tensor_names.push(name);
    }

    let architecture = string_metadata(&metadata, "general.architecture");
    let layer_count = architecture
        .as_deref()
        .and_then(|architecture| u64_metadata(&metadata, &format!("{architecture}.block_count")))
        .and_then(|value| u32::try_from(value).ok());
    let hidden_size = architecture
        .as_deref()
        .and_then(|architecture| {
            u64_metadata(&metadata, &format!("{architecture}.embedding_length"))
        })
        .and_then(|value| usize::try_from(value).ok());
    let tokenizer_family = string_metadata(&metadata, "tokenizer.ggml.model");
    let quantization =
        u64_metadata(&metadata, "general.file_type").map(|value| format!("gguf_file_type_{value}"));
    let tokenizer_checksum = tokenizer_checksum(&metadata);

    Ok(GgufInfo {
        version,
        metadata_kv_count,
        tensor_count,
        architecture,
        layer_count,
        hidden_size,
        tokenizer_family,
        tokenizer_checksum,
        quantization,
        tensor_names,
    })
}

pub fn sha256_file(path: &Path) -> Result<String> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let mut reader = BufReader::new(file);
    let mut hasher = Sha256::new();
    let mut buffer = [0; 1024 * 64];

    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }

    Ok(hex_lower(&hasher.finalize()))
}

fn tokenizer_checksum(metadata: &BTreeMap<String, GgufValue>) -> String {
    let mut hasher = Sha256::new();

    for (key, value) in metadata
        .iter()
        .filter(|(key, _)| key.starts_with("tokenizer."))
    {
        hasher.update(key.as_bytes());
        hasher.update([0]);
        hasher.update(value.to_string().as_bytes());
        hasher.update([0xff]);
    }

    hex_lower(&hasher.finalize())
}

fn string_metadata(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<String> {
    metadata.get(key).and_then(GgufValue::as_string)
}

fn u64_metadata(metadata: &BTreeMap<String, GgufValue>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(GgufValue::as_u64)
}

fn read_value(reader: &mut impl Read, value_type: u32) -> Result<GgufValue> {
    match value_type {
        0 => Ok(GgufValue::UInt(u64::from(read_u8(reader)?))),
        1 => Ok(GgufValue::Int(i64::from(read_i8(reader)?))),
        2 => Ok(GgufValue::UInt(u64::from(read_u16(reader)?))),
        3 => Ok(GgufValue::Int(i64::from(read_i16(reader)?))),
        4 => Ok(GgufValue::UInt(u64::from(read_u32(reader)?))),
        5 => Ok(GgufValue::Int(i64::from(read_i32(reader)?))),
        6 => Ok(GgufValue::Float(read_f32(reader)?.to_string())),
        7 => Ok(GgufValue::Bool(read_bool(reader)?)),
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => {
            let element_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            let mut hasher = Sha256::new();
            hasher.update(element_type.to_le_bytes());
            hasher.update(len.to_le_bytes());

            for _ in 0..len {
                hash_value(reader, element_type, &mut hasher)?;
            }

            Ok(GgufValue::Array {
                element_type,
                len,
                checksum_sha256: hex_lower(&hasher.finalize()),
            })
        }
        10 => Ok(GgufValue::UInt(read_u64(reader)?)),
        11 => Ok(GgufValue::Int(read_i64(reader)?)),
        12 => Ok(GgufValue::Float(read_f64(reader)?.to_string())),
        other => Err(anyhow!("unsupported GGUF value type {other}")),
    }
}

fn hash_value(reader: &mut impl Read, value_type: u32, hasher: &mut Sha256) -> Result<()> {
    match value_type {
        0 => hasher.update([read_u8(reader)?]),
        1 => hasher.update(read_i8(reader)?.to_le_bytes()),
        2 => hasher.update(read_u16(reader)?.to_le_bytes()),
        3 => hasher.update(read_i16(reader)?.to_le_bytes()),
        4 => hasher.update(read_u32(reader)?.to_le_bytes()),
        5 => hasher.update(read_i32(reader)?.to_le_bytes()),
        6 => hasher.update(read_f32(reader)?.to_le_bytes()),
        7 => hasher.update([u8::from(read_bool(reader)?)]),
        8 => {
            let value = read_string_bytes(reader)?;
            hasher.update((value.len() as u64).to_le_bytes());
            hasher.update(value);
        }
        9 => {
            let element_type = read_u32(reader)?;
            let len = read_u64(reader)?;
            hasher.update(element_type.to_le_bytes());
            hasher.update(len.to_le_bytes());
            for _ in 0..len {
                hash_value(reader, element_type, hasher)?;
            }
        }
        10 => hasher.update(read_u64(reader)?.to_le_bytes()),
        11 => hasher.update(read_i64(reader)?.to_le_bytes()),
        12 => hasher.update(read_f64(reader)?.to_le_bytes()),
        other => bail!("unsupported GGUF array value type {other}"),
    }

    Ok(())
}

fn read_string(reader: &mut impl Read) -> Result<String> {
    let bytes = read_string_bytes(reader)?;
    String::from_utf8(bytes).context("GGUF string is not valid UTF-8")
}

fn read_string_bytes(reader: &mut impl Read) -> Result<Vec<u8>> {
    let len = read_u64(reader)?;
    let len = usize::try_from(len).context("GGUF string length does not fit in memory")?;
    let mut bytes = vec![0; len];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u8(reader: &mut impl Read) -> io::Result<u8> {
    let mut bytes = [0; 1];
    reader.read_exact(&mut bytes)?;
    Ok(bytes[0])
}

fn read_i8(reader: &mut impl Read) -> io::Result<i8> {
    Ok(read_u8(reader)? as i8)
}

fn read_bool(reader: &mut impl Read) -> io::Result<bool> {
    Ok(read_u8(reader)? != 0)
}

fn read_u16(reader: &mut impl Read) -> io::Result<u16> {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes)?;
    Ok(u16::from_le_bytes(bytes))
}

fn read_i16(reader: &mut impl Read) -> io::Result<i16> {
    let mut bytes = [0; 2];
    reader.read_exact(&mut bytes)?;
    Ok(i16::from_le_bytes(bytes))
}

fn read_u32(reader: &mut impl Read) -> io::Result<u32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(u32::from_le_bytes(bytes))
}

fn read_i32(reader: &mut impl Read) -> io::Result<i32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(i32::from_le_bytes(bytes))
}

fn read_f32(reader: &mut impl Read) -> io::Result<f32> {
    let mut bytes = [0; 4];
    reader.read_exact(&mut bytes)?;
    Ok(f32::from_le_bytes(bytes))
}

fn read_u64(reader: &mut impl Read) -> io::Result<u64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(u64::from_le_bytes(bytes))
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

fn read_f64(reader: &mut impl Read) -> io::Result<f64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(f64::from_le_bytes(bytes))
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);

    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }

    output
}
