use infernet_model::LayerRange;
use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum RuntimeError {
    #[error("payload hidden size {actual} does not match runtime hidden size {expected}")]
    HiddenSizeMismatch { actual: usize, expected: usize },
    #[error("requested layer range {requested:?} is outside owned range {owned:?}")]
    RangeNotOwned {
        requested: LayerRange,
        owned: LayerRange,
    },
}

pub trait LayerRuntime {
    fn owned_layers(&self) -> LayerRange;
    fn hidden_size(&self) -> usize;
    fn execute(&self, layers: LayerRange, payload: &[f32]) -> Result<Vec<f32>, RuntimeError>;
}

#[derive(Debug, Clone)]
pub struct DemoRuntime {
    owned_layers: LayerRange,
    hidden_size: usize,
}

impl DemoRuntime {
    pub fn new(owned_layers: LayerRange, hidden_size: usize) -> Self {
        Self {
            owned_layers,
            hidden_size,
        }
    }

    pub fn prompt_to_activation(prompt: &str, hidden_size: usize) -> Vec<f32> {
        let mut activation = vec![0.0; hidden_size];

        if prompt.is_empty() {
            return activation;
        }

        for (index, byte) in prompt.bytes().enumerate() {
            let slot = index % hidden_size;
            let scaled = f32::from(byte) / 255.0;
            activation[slot] += scaled * (1.0 + (index % 7) as f32 * 0.03);
        }

        let divisor = prompt.len().max(1) as f32;
        for value in &mut activation {
            *value = (*value / divisor).tanh();
        }

        activation
    }

    pub fn decode_activation(payload: &[f32]) -> String {
        let checksum = activation_checksum(payload);
        format!("infernet-demo-{checksum:016x}")
    }

    fn execute_layer(layer: u32, input: &[f32]) -> Vec<f32> {
        let len = input.len();
        let layer_scale = 1.0 + layer as f32 * 0.013;
        let layer_bias = (layer + 1) as f32 * 0.001;

        (0..len)
            .map(|index| {
                let left = input[(index + len - 1) % len];
                let center = input[index];
                let right = input[(index + 1) % len];
                (center * layer_scale + left * 0.05 - right * 0.025 + layer_bias).tanh()
            })
            .collect()
    }
}

impl LayerRuntime for DemoRuntime {
    fn owned_layers(&self) -> LayerRange {
        self.owned_layers
    }

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn execute(&self, layers: LayerRange, payload: &[f32]) -> Result<Vec<f32>, RuntimeError> {
        if payload.len() != self.hidden_size {
            return Err(RuntimeError::HiddenSizeMismatch {
                actual: payload.len(),
                expected: self.hidden_size,
            });
        }

        if !self.owned_layers.contains(&layers) {
            return Err(RuntimeError::RangeNotOwned {
                requested: layers,
                owned: self.owned_layers,
            });
        }

        let mut current = payload.to_vec();

        for layer in layers.start..layers.end {
            current = Self::execute_layer(layer, &current);
        }

        Ok(current)
    }
}

pub fn activation_checksum(payload: &[f32]) -> u64 {
    payload
        .iter()
        .enumerate()
        .fold(0xcbf29ce484222325, |hash, (index, value)| {
            let bits = value.to_bits() as u64;
            let mixed = bits ^ ((index as u64 + 1) * 0x100000001b3);
            hash.wrapping_mul(0x100000001b3) ^ mixed
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_execution_matches_full_execution() {
        let prompt = "distributed inference";
        let hidden_size = 16;
        let input = DemoRuntime::prompt_to_activation(prompt, hidden_size);
        let full = DemoRuntime::new(LayerRange::new(0, 12).unwrap(), hidden_size)
            .execute(LayerRange::new(0, 12).unwrap(), &input)
            .unwrap();

        let ranges = [
            LayerRange::new(0, 3).unwrap(),
            LayerRange::new(3, 6).unwrap(),
            LayerRange::new(6, 9).unwrap(),
            LayerRange::new(9, 12).unwrap(),
        ];

        let mut split = input;
        for range in ranges {
            split = DemoRuntime::new(range, hidden_size)
                .execute(range, &split)
                .unwrap();
        }

        assert_eq!(split, full);
    }

    #[test]
    fn runtime_rejects_unowned_layers() {
        let runtime = DemoRuntime::new(LayerRange::new(0, 3).unwrap(), 16);
        let payload = vec![0.0; 16];
        let err = runtime
            .execute(LayerRange::new(3, 6).unwrap(), &payload)
            .unwrap_err();

        assert_eq!(
            err,
            RuntimeError::RangeNotOwned {
                requested: LayerRange::new(3, 6).unwrap(),
                owned: LayerRange::new(0, 3).unwrap()
            }
        );
    }
}
