use expander_compiler::frontend::{Config, Define, RootAPI, Variable};
use expander_compiler::frontend::internal::DumpLoadTwoVariables;

struct QuantizedFloat(Variable);

impl QuantizedFloat {
    /// Circuit for add two quantized values
    fn add<C: Config, B: RootAPI<C>>(&self, api: &mut B, b: &Self) -> Self {
        QuantizedFloat(api.add(self.0, b.0))
    }

    /// Circuit for multiplying two quantized values
    fn mul<C: Config, B: RootAPI<C>>(&self, api: &mut B, b: &Self, scale_inv: Variable) -> Self {
        // multiply into accumulator
        let acc_mul = api.mul(self.0, b.0);
        // rescale
        let rescaled_mul = api.mul(acc_mul, scale_inv);
        QuantizedFloat(rescaled_mul)
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::quantized_float::QuantizedFloat;
    use crate::quantization::quantizer::Quantizer;
    use expander_compiler::compile::CompileOptions;
    use expander_compiler::declare_circuit;
    use expander_compiler::field::BN254;
    use expander_compiler::frontend::{BN254Config, compile, Define, RootAPI, Variable};

    #[test]
    fn test_quantized_add() {
        declare_circuit!(AddCircuit {
            a: Variable,
            b: Variable,
            target: PublicVariable
        });

        impl Define<BN254Config> for AddCircuit<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let a = QuantizedFloat(self.a);
                let b = QuantizedFloat(self.b);
                let sum = a.add(api, &b);
                api.assert_is_equal(sum.0, self.target);
            }
        }

        const N: u8 = 2;
        let q = Quantizer::<N> {};
        let compile_result = compile(&AddCircuit::default(), CompileOptions::default()).unwrap();
        // TODO: make this a more complicated assignment
        let assignment = AddCircuit::<BN254> {
            a: q.quantize(5.),
            b: q.quantize(-2.),
            target: q.quantize(3.)
        };
        let witness = compile_result.witness_solver.solve_witness(&assignment).unwrap();
        let run_result = compile_result.layered_circuit.run(&witness);
        assert!(run_result.iter().all(|v| *v));
    }

    // #[test]
    // fn test_circuit() {
    //     const N: u8 = 1;
    //     let q = Quantizer::<N> {};
    //     let compile_result = compile(&TestCircuit::default(), CompileOptions::default()).unwrap();
    //     let assignment = TestCircuit::<BN254> {
    //         a: q.quantize(3.5),
    //         b: q.quantize(3.0),
    //         scale_inv: BN254::from(1_u32 << N).inv().unwrap(),
    //         target: q.quantize(10.5),
    //     };
    //
    //     let witness = compile_result
    //         .witness_solver
    //         .solve_witness(&assignment)
    //         .unwrap();
    //     let run_result = compile_result.layered_circuit.run(&witness);
    //     dbg!(run_result);
    //     debug_eval(&TestCircuit::default(), &assignment, EmptyHintCaller)
    // }
}
