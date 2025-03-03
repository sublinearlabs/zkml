use expander_compiler::field::FieldArith;
use expander_compiler::field::BN254;

// TODO: implement a type that hides all the quantization logic

// TODO: forced to use BN254 for now, goldilocks would have been preferred
//  might have to look into using multiple limbs of m31 or reducing the
//  fixed point space to fit into 15 / 16 bits

// TODO: add documentation
fn f32_to_field(float_value: f32, fractional_bit_count: u32) -> BN254 {
    let mut scaled_float = float_value * (1 << fractional_bit_count) as f32;
    if float_value >= 0. {
        scaled_float += 0.5
    } else {
        scaled_float -= 0.5
    }
    let fixed_rep = (scaled_float as i32) as u32;
    BN254::from(fixed_rep)
}

// TODO: add documentation
fn field_to_f32(field_value: &BN254, fractional_bit_count: u32) -> f32 {
    let mut lower_u32 = [0; 4];
    lower_u32.copy_from_slice(&field_value.to_bytes()[0..4]);
    u32::from_le_bytes(lower_u32) as f32 / (1 << fractional_bit_count) as f32
}

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
    use crate::{f32_to_field, field_to_f32, generate_granularity_steps};
    use expander_compiler::frontend::BN254;

    #[test]
    fn test_i32_as_u32_same_bit_pattern() {
        let a: i32 = -6;
        let b = a as u32;
        assert_eq!(format!("{:b}", a), format!("{:b}", b));
    }

    #[test]
    fn test_f32_to_and_from_field() {
        // using precise granularity range for easy testing
        assert_eq!(field_to_f32(&f32_to_field(3.5, 2), 2) - 3.5, 0.0);
        assert_eq!(
            field_to_f32(&f32_to_field(485.68359375, 8), 8) - 485.68359375,
            0.0
        );
        assert_eq!(
            field_to_f32(&f32_to_field(32767.99884033, 16), 16) - 32767.99884033,
            0.0
        );
    }
}
