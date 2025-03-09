use expander_compiler::frontend::internal::DumpLoadTwoVariables;
use expander_compiler::frontend::{Config, Define, RootAPI, Variable};

#[derive(Default, Debug, Clone, Copy)]
pub(crate) struct QuantizedFloat(Variable);

impl QuantizedFloat {
    /// Instantiate a new quantized float from a variable
    pub fn new(var: Variable) -> Self {
        Self(var)
    }

    /// Circuit for add two quantized values
    pub(crate) fn add<C: Config, B: RootAPI<C>>(&self, api: &mut B, b: &Self) -> Self {
        QuantizedFloat(api.add(self.0, b.0))
    }

    /// Circuit for multiplying two quantized values
    pub(crate) fn mul<C: Config, B: RootAPI<C>>(
        &self,
        api: &mut B,
        b: &Self,
        scale_inv: Variable,
    ) -> Self {
        // multiply into accumulator
        let acc_mul = api.mul(self.0, b.0);
        // rescale
        let rescaled_mul = api.mul(acc_mul, scale_inv);
        QuantizedFloat(rescaled_mul)
    }

    /// Return inner variable
    pub(crate) fn to_var(self) -> Variable {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use crate::quantization::quantized_float::QuantizedFloat;
    use crate::quantization::quantizer::Quantizer;
    use expander_compiler::compile::CompileOptions;
    use expander_compiler::declare_circuit;
    use expander_compiler::field::{FieldArith, BN254};
    use expander_compiler::frontend::{compile, BN254Config, Define, RootAPI, Variable};

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

        const N: u8 = 8;
        let q = Quantizer::<N> {};
        let compile_result = compile(&AddCircuit::default(), CompileOptions::default()).unwrap();
        let assignment = AddCircuit::<BN254> {
            a: q.quantize(-26.625),
            b: q.quantize(40.25),
            target: q.quantize(13.625),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let run_result = compile_result.layered_circuit.run(&witness);
        assert!(run_result.iter().all(|v| *v));
    }

    #[test]
    fn test_quantized_mul() {
        declare_circuit!(MulCircuit {
            a: Variable,
            b: Variable,
            scale_inv: PublicVariable,
            target: PublicVariable
        });

        impl Define<BN254Config> for MulCircuit<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let a = QuantizedFloat(self.a);
                let b = QuantizedFloat(self.b);
                let sum = a.mul(api, &b, self.scale_inv);
                api.assert_is_equal(sum.0, self.target);
            }
        }

        const N: u8 = 8;
        let q = Quantizer::<N> {};
        let compile_result = compile(&MulCircuit::default(), CompileOptions::default()).unwrap();
        let assignment = MulCircuit::<BN254> {
            a: q.quantize(26.5),
            b: q.quantize(-40.25),
            scale_inv: BN254::from(q.scale()).inv().unwrap(),
            target: q.quantize(-1066.625),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let run_result = compile_result.layered_circuit.run(&witness);
        assert!(run_result.iter().all(|v| *v));
    }
}
