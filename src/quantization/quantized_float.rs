use expander_compiler::declare_circuit;
use expander_compiler::frontend::{BN254Config, Config, Define, RootAPI, Variable};

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
mod tests {
    use crate::quantization::quantized_float::TestCircuit;
    use crate::quantization::quantizer::Quantizer;
    use expander_compiler::compile::CompileOptions;
    use expander_compiler::field::{FieldArith, BN254};
    use expander_compiler::frontend::extra::debug_eval;
    use expander_compiler::frontend::{compile, EmptyHintCaller};

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
