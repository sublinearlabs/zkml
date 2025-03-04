use expander_compiler::field::FieldArith;
use expander_compiler::field::BN254;

// TODO: forced to use BN254 for now, goldilocks would have been preferred
//  might have to look into using multiple limbs of m31 or reducing the
//  fixed point space to fit into 15 / 16 bits

// TODO: document N
struct Quantizer<const N: u32> {}

impl<const N: u32> Quantizer<N> {
    // TODO: documentation / replace BN254
    fn quantize(&self, value: f32) -> BN254 {
        let mut scaled_float = value * (1 << N) as f32;
        if value >= 0. {
            scaled_float += 0.5
        } else {
            scaled_float -= 0.5
        }
        let fixed_rep = (scaled_float as i32) as u32;
        BN254::from(fixed_rep)
    }

    // TODO: add documentation
    fn dequantize(&self, value: &BN254) -> f32 {
        let mut lower_u32 = [0; 4];
        lower_u32.copy_from_slice(&value.to_bytes()[0..4]);
        u32::from_le_bytes(lower_u32) as f32 / (1 << N) as f32
    }
}

// TODO: add documentation
fn generate_granularity_steps(fractional_bit_count: u32) {
    let granularity = 1.0 / (1 << fractional_bit_count) as f32;
    let mut curr = 0.0;
    while curr != 1.0 {
        println!("{}", curr);
        curr += granularity
    }
}

#[cfg(test)]
mod test {
    use crate::Quantizer;

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
        let v = 3.5;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);

        let q = Quantizer::<8> {};
        let v = 485.68359375;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);

        let q = Quantizer::<16> {};
        let v = 32767.99884033;
        assert_eq!(q.dequantize(&q.quantize(v)) - v, 0.0);
    }
}
