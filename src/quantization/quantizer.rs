use expander_compiler::field::BN254;

/// N represents the number of fractional bits for the fixed point representation
pub struct Quantizer<const N: u8> {}

impl<const N: u8> Quantizer<N> {
    /// Converts an f32 value into signed fixed point representation and represent that
    /// value in a field
    pub fn quantize(&self, value: f32) -> BN254 {
        let mut scaled_float = value * (1 << N) as f32;

        // to minimize quantization error from granularity g to g/2
        // we add \pm 0.5 to bias value to the closest fixed point rep (rounding)
        // as opposed to just direct truncation
        if value >= 0. {
            scaled_float += 0.5
        } else {
            scaled_float -= 0.5
        }

        // first convert to i32 to handle sign extension then convert to u32
        // bit pattern is preserved see `test_i32_as_u32_same_bit_pattern`
        let fixed_rep = (scaled_float as i32) as u32;

        BN254::from(fixed_rep)
    }

    /// Converts a field representation of a fixed point value to the equivalent f32 value
    pub fn dequantize(&self, value: &BN254) -> f32 {
        let mut lower_u32 = [0; 4];
        lower_u32.copy_from_slice(&value.to_bytes()[0..4]);
        i32::from_le_bytes(lower_u32) as f32 / (1 << N) as f32
    }

    pub fn scale(&self) -> u32 {
        1 << N
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::quantizer::Quantizer;
    use expander_compiler::field::{FieldArith, BN254};

    #[test]
    fn test_i32_as_u32_same_bit_pattern() {
        let a: i32 = -6;
        let b = a as u32;
        assert_eq!(format!("{:b}", a), format!("{:b}", b));
    }

    #[test]
    fn test_f32_to_and_from_field() {
        // using precise granularity range for easy testing
        let q = Quantizer::<2> {};
        let v = -3.5;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);

        let q = Quantizer::<8> {};
        let v = 485.68359375;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);

        let q = Quantizer::<16> {};
        let v = 32767.99884033;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);
    }
}
