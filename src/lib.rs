use expander_compiler::declare_circuit;
use expander_compiler::field::FieldArith;
use expander_compiler::field::BN254;
use expander_compiler::frontend::{BN254Config, Config, Define, M31Config, RootAPI, Variable};

// TODO: forced to use BN254 for now, goldilocks would have been preferred
//  might have to look into using multiple limbs of m31 or reducing the
//  fixed point space to fit into 15 / 16 bits

/// N represents the number of fractional bits for the fixed point representation
struct Quantizer<const N: u8> {}

impl<const N: u8> Quantizer<N> {
    /// Converts an f32 value into signed fixed point representation and represent that
    /// value in a field
    fn quantize(&self, value: f32) -> BN254 {
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
    fn dequantize(&self, value: &BN254) -> f32 {
        let mut lower_u32 = [0; 4];
        lower_u32.copy_from_slice(&value.to_bytes()[0..4]);
        u32::from_le_bytes(lower_u32) as f32 / (1 << N) as f32
    }
}

// TODO: add documentation
fn add_q<C: Config, B: RootAPI<C>>(api: &mut B, a: Variable, b: Variable) -> Variable {
    api.add(a, b)
}

// TODO: add documentation
fn mul_q<C: Config, B: RootAPI<C>>(
    api: &mut B,
    a: Variable,
    b: Variable,
    scale_inv: Variable,
) -> Variable {
    // we need first multiply then rescale
    let accumulated_mul = api.mul(a, b);
    api.display("acc_mul", accumulated_mul);
    // TODO: what is checked??
    api.mul(accumulated_mul, scale_inv)
}

declare_circuit!(TestCircuit {
    a: Variable,
    b: Variable,
    scale_inv: Variable,
    target: PublicVariable
});

impl Define<BN254Config> for TestCircuit<Variable> {
    fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
        // let sum = add_q(api, self.a, self.b);
        api.display("a", self.a);
        api.display("b", self.b);
        let prod = mul_q(api, self.a, self.b, self.scale_inv);
        api.display("product", prod);
        api.display("target", self.target);
        api.assert_is_equal(prod, self.target);
    }
}

#[cfg(test)]
mod test {
    use crate::{Quantizer, TestCircuit};
    use expander_compiler::compile::CompileOptions;
    use expander_compiler::field::{FieldArith, BN254};
    use expander_compiler::frontend::extra::debug_eval;
    use expander_compiler::frontend::{compile, BN254Config, EmptyHintCaller};

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

    #[test]
    fn test_circuit() {
        const N: u8 = 1;
        let q = Quantizer::<N> {};
        let compile_result = compile(&TestCircuit::default(), CompileOptions::default()).unwrap();
        let assignment = TestCircuit::<BN254> {
            a: q.quantize(3.5),
            b: q.quantize(3.0),
            scale_inv: BN254::from(1_u32 << N).inv().unwrap(),
            target: q.quantize(10.5),
        };

        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let run_result = compile_result.layered_circuit.run(&witness);
        dbg!(run_result);
        debug_eval(&TestCircuit::default(), &assignment, EmptyHintCaller)
    }
}
