use expander_compiler::field::BN254;

// TODO: add documentation
fn f32_to_field(float_value: f32, fractional_bit_count: usize) -> BN254 {
    let mut scaled_float = float_value * (1 << fractional_bit_count) as f32;
    if float_value >= 0. {
        scaled_float += 0.5
    } else {
        scaled_float -= 0.5
    }
    let fixed_rep = (scaled_float as i32) as u32;
    BN254::from(fixed_rep)
}

#[cfg(test)]
mod test {
    #[test]
    fn test_i32_as_u32_same_bit_pattern() {
        let a: i32 = -6;
        let b = a as u32;
        assert_eq!(
            format!("{:b}", a),
            format!("{:b}", b)
        );
    }
}