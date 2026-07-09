use std::{
    collections::BTreeMap,
    fmt,
    fs::File,
    io::{self, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
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
pub struct GgufLayerShardSummary {
    pub path: PathBuf,
    pub tensor_names: Vec<String>,
    pub size_bytes: u64,
}

#[derive(Debug, Clone)]
struct TensorInfo {
    name: String,
    dimensions: Vec<u64>,
    tensor_type: u32,
    offset: u64,
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
    let quantization = u64_metadata(&metadata, "general.file_type").map(gguf_file_type_name);
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

pub fn write_layer_shard(
    source: &Path,
    destination: &Path,
    layers: crate::LayerRange,
    layer_count: u32,
) -> Result<GgufLayerShardSummary> {
    layers.validate_for_model(layer_count)?;

    let parsed = parse_gguf_layout(source)?;
    let kept_tensors = parsed
        .tensors
        .iter()
        .filter(|tensor| tensor_required_for_layers(&tensor.name, layers))
        .collect::<Vec<_>>();

    if kept_tensors.is_empty() {
        bail!(
            "GGUF shard {}:{} selected no tensors from {}",
            layers.start,
            layers.end,
            source.display()
        );
    }

    if let Some(parent) = destination.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }

    let mut source_file =
        File::open(source).with_context(|| format!("failed to open {}", source.display()))?;
    let mut output = File::create(destination)
        .with_context(|| format!("failed to create {}", destination.display()))?;

    let mut tensor_spans = BTreeMap::<u64, u64>::new();
    let mut offsets = parsed
        .tensors
        .iter()
        .map(|tensor| tensor.offset)
        .collect::<Vec<_>>();
    offsets.sort_unstable();
    offsets.dedup();
    for (index, offset) in offsets.iter().enumerate() {
        let next = offsets
            .get(index + 1)
            .copied()
            .unwrap_or(parsed.file_size.saturating_sub(parsed.data_start));
        if next < *offset {
            bail!("GGUF tensor offsets are not sorted in {}", source.display());
        }
        tensor_spans.insert(*offset, next - *offset);
    }

    let mut new_offsets = BTreeMap::<String, u64>::new();
    let mut next_data_offset = 0_u64;
    for tensor in &kept_tensors {
        next_data_offset = align_up(next_data_offset, parsed.alignment)?;
        new_offsets.insert(tensor.name.clone(), next_data_offset);
        let span = tensor_spans
            .get(&tensor.offset)
            .copied()
            .ok_or_else(|| anyhow!("missing span for tensor {}", tensor.name))?;
        next_data_offset = next_data_offset
            .checked_add(span)
            .ok_or_else(|| anyhow!("GGUF shard data offset overflow"))?;
    }

    output.write_all(b"GGUF")?;
    write_u32(&mut output, parsed.version)?;
    write_u64(&mut output, kept_tensors.len() as u64)?;
    write_u64(&mut output, parsed.metadata_kv_count)?;
    output.write_all(&parsed.metadata_bytes)?;
    for tensor in &kept_tensors {
        write_string(&mut output, &tensor.name)?;
        write_u32(&mut output, tensor.dimensions.len() as u32)?;
        for dimension in &tensor.dimensions {
            write_u64(&mut output, *dimension)?;
        }
        write_u32(&mut output, tensor.tensor_type)?;
        write_u64(
            &mut output,
            *new_offsets
                .get(&tensor.name)
                .expect("new offset inserted before writing"),
        )?;
    }

    let header_end = output.stream_position()?;
    let new_data_start = align_up(header_end, parsed.alignment)?;
    write_zero_padding(&mut output, new_data_start - header_end)?;

    for tensor in &kept_tensors {
        let new_offset = *new_offsets
            .get(&tensor.name)
            .expect("new offset inserted before copying");
        let target_position = new_data_start
            .checked_add(new_offset)
            .ok_or_else(|| anyhow!("GGUF shard output offset overflow"))?;
        let current_position = output.stream_position()?;
        if current_position > target_position {
            bail!(
                "GGUF shard writer overran tensor offset for {}; current={}, target={}",
                tensor.name,
                current_position,
                target_position
            );
        }
        write_zero_padding(&mut output, target_position - current_position)?;

        let old_position = parsed
            .data_start
            .checked_add(tensor.offset)
            .ok_or_else(|| anyhow!("GGUF source tensor offset overflow"))?;
        let span = tensor_spans
            .get(&tensor.offset)
            .copied()
            .ok_or_else(|| anyhow!("missing span for tensor {}", tensor.name))?;
        copy_range(&mut source_file, &mut output, old_position, span)
            .with_context(|| format!("failed to copy tensor {}", tensor.name))?;
    }

    output.flush()?;
    let size_bytes = output.metadata()?.len();
    Ok(GgufLayerShardSummary {
        path: destination.to_path_buf(),
        tensor_names: kept_tensors
            .into_iter()
            .map(|tensor| tensor.name.clone())
            .collect(),
        size_bytes,
    })
}

#[derive(Debug, Clone)]
struct GgufLayout {
    version: u32,
    metadata_kv_count: u64,
    metadata_bytes: Vec<u8>,
    tensors: Vec<TensorInfo>,
    alignment: u64,
    data_start: u64,
    file_size: u64,
}

fn parse_gguf_layout(path: &Path) -> Result<GgufLayout> {
    let mut file =
        File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    let file_size = file.metadata()?.len();

    let mut magic = [0; 4];
    file.read_exact(&mut magic)?;
    if &magic != b"GGUF" {
        bail!("{} is not a GGUF file", path.display());
    }

    let version = read_u32(&mut file)?;
    let tensor_count = read_u64(&mut file)?;
    let metadata_kv_count = read_u64(&mut file)?;
    let metadata_start = file.stream_position()?;
    let mut alignment = 32_u64;

    for _ in 0..metadata_kv_count {
        let key = read_string(&mut file)?;
        let value_type = read_u32(&mut file)?;
        let value = read_value(&mut file, value_type)
            .with_context(|| format!("failed to read GGUF metadata value {key}"))?;
        if key == "general.alignment" {
            alignment = value.as_u64().unwrap_or(alignment).max(1);
        }
    }

    let metadata_end = file.stream_position()?;
    file.seek(SeekFrom::Start(metadata_start))?;
    let metadata_len = usize::try_from(metadata_end - metadata_start)
        .context("GGUF metadata is too large to fit in memory")?;
    let mut metadata_bytes = vec![0; metadata_len];
    file.read_exact(&mut metadata_bytes)?;
    file.seek(SeekFrom::Start(metadata_end))?;

    let mut tensors = Vec::with_capacity(tensor_count.min(usize::MAX as u64) as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let dimensions_len = read_u32(&mut file)?;
        let mut dimensions = Vec::with_capacity(dimensions_len as usize);
        for _ in 0..dimensions_len {
            dimensions.push(read_u64(&mut file)?);
        }
        let tensor_type = read_u32(&mut file)?;
        let offset = read_u64(&mut file)?;
        tensors.push(TensorInfo {
            name,
            dimensions,
            tensor_type,
            offset,
        });
    }

    let tensor_info_end = file.stream_position()?;
    let data_start = align_up(tensor_info_end, alignment)?;
    if data_start > file_size {
        bail!("GGUF data section starts past EOF in {}", path.display());
    }

    Ok(GgufLayout {
        version,
        metadata_kv_count,
        metadata_bytes,
        tensors,
        alignment,
        data_start,
        file_size,
    })
}

fn tensor_required_for_layers(name: &str, layers: crate::LayerRange) -> bool {
    match tensor_layer_index(name) {
        Some(layer) => layers.start <= layer && layer < layers.end,
        None => true,
    }
}

fn tensor_layer_index(name: &str) -> Option<u32> {
    let rest = name.strip_prefix("blk.")?;
    let (layer, _) = rest.split_once('.')?;
    layer.parse::<u32>().ok()
}

fn align_up(value: u64, alignment: u64) -> Result<u64> {
    if alignment == 0 {
        bail!("alignment must be greater than zero");
    }
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| anyhow!("alignment overflow"))
    }
}

fn copy_range(input: &mut File, output: &mut File, offset: u64, len: u64) -> Result<()> {
    input.seek(SeekFrom::Start(offset))?;
    let mut remaining = len;
    let mut buffer = [0; 1024 * 1024];
    while remaining > 0 {
        let read_len = buffer.len().min(remaining as usize);
        let read = input.read(&mut buffer[..read_len])?;
        if read == 0 {
            bail!("unexpected EOF while copying GGUF tensor data");
        }
        output.write_all(&buffer[..read])?;
        remaining -= read as u64;
    }
    Ok(())
}

fn write_zero_padding(output: &mut File, len: u64) -> io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    const ZEROES: [u8; 4096] = [0; 4096];
    let mut remaining = len;
    while remaining > 0 {
        let write_len = ZEROES.len().min(remaining as usize);
        output.write_all(&ZEROES[..write_len])?;
        remaining -= write_len as u64;
    }
    Ok(())
}

fn gguf_file_type_name(value: u64) -> String {
    match value {
        0 => "F32",
        1 => "F16",
        2 => "Q4_0",
        3 => "Q4_1",
        7 => "Q8_0",
        8 => "Q5_0",
        9 => "Q5_1",
        10 => "Q2_K",
        11 => "Q3_K_S",
        12 => "Q3_K_M",
        13 => "Q3_K_L",
        14 => "Q4_K_S",
        15 => "Q4_K_M",
        16 => "Q5_K_S",
        17 => "Q5_K_M",
        18 => "Q6_K",
        19 => "IQ2_XXS",
        20 => "IQ2_XS",
        21 => "Q2_K_S",
        22 => "IQ3_XS",
        23 => "IQ3_XXS",
        24 => "IQ1_S",
        25 => "IQ4_NL",
        26 => "IQ3_S",
        27 => "IQ3_M",
        28 => "IQ2_S",
        29 => "IQ2_M",
        30 => "IQ4_XS",
        31 => "IQ1_M",
        32 => "BF16",
        36 => "TQ1_0",
        37 => "TQ2_0",
        38 => "MXFP4_MOE",
        39 => "NVFP4",
        40 => "Q1_0",
        41 => "Q2_0",
        _ => return format!("gguf_file_type_{value}"),
    }
    .to_owned()
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

fn write_string(writer: &mut impl Write, value: &str) -> io::Result<()> {
    write_u64(writer, value.len() as u64)?;
    writer.write_all(value.as_bytes())
}

fn write_u32(writer: &mut impl Write, value: u32) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn write_u64(writer: &mut impl Write, value: u64) -> io::Result<()> {
    writer.write_all(&value.to_le_bytes())
}

fn read_i64(reader: &mut impl Read) -> io::Result<i64> {
    let mut bytes = [0; 8];
    reader.read_exact(&mut bytes)?;
    Ok(i64::from_le_bytes(bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LayerRange;
    use std::fs;

    #[test]
    fn writes_layer_shard_with_only_selected_block_tensors() {
        let root = std::env::temp_dir().join(format!(
            "infernet-gguf-shard-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&root).unwrap();
        let source = root.join("source.gguf");
        let shard = root.join("shard.gguf");
        write_test_gguf(&source).unwrap();

        let summary =
            write_layer_shard(&source, &shard, LayerRange::new(1, 2).unwrap(), 3).unwrap();
        let info = parse_gguf_info(&shard).unwrap();

        assert_eq!(summary.size_bytes, fs::metadata(&shard).unwrap().len());
        assert_eq!(
            info.tensor_names,
            vec![
                "token_embd.weight".to_owned(),
                "blk.1.attn_norm.weight".to_owned(),
                "output_norm.weight".to_owned()
            ]
        );

        let _ = fs::remove_dir_all(root);
    }

    fn write_test_gguf(path: &Path) -> Result<()> {
        let mut output = File::create(path)?;
        output.write_all(b"GGUF")?;
        write_u32(&mut output, 3)?;
        write_u64(&mut output, 5)?;
        write_u64(&mut output, 4)?;

        write_string(&mut output, "general.architecture")?;
        write_u32(&mut output, 8)?;
        write_string(&mut output, "llama")?;
        write_string(&mut output, "llama.block_count")?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, 3)?;
        write_string(&mut output, "llama.embedding_length")?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, 1)?;
        write_string(&mut output, "general.alignment")?;
        write_u32(&mut output, 4)?;
        write_u32(&mut output, 32)?;

        let tensors = [
            ("token_embd.weight", 0_u64),
            ("blk.0.attn_norm.weight", 32),
            ("blk.1.attn_norm.weight", 64),
            ("blk.2.attn_norm.weight", 96),
            ("output_norm.weight", 128),
        ];
        for (name, offset) in tensors {
            write_string(&mut output, name)?;
            write_u32(&mut output, 1)?;
            write_u64(&mut output, 1)?;
            write_u32(&mut output, 0)?;
            write_u64(&mut output, offset)?;
        }

        let header_end = output.stream_position()?;
        let data_start = align_up(header_end, 32)?;
        write_zero_padding(&mut output, data_start - header_end)?;
        for value in 0_u8..5 {
            output.write_all(&[value; 4])?;
            write_zero_padding(&mut output, 28)?;
        }

        Ok(())
    }
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
