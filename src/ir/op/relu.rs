use std::collections::HashMap;

use expander_compiler::{
    field::BN254,
    frontend::{Config, Error, RootAPI, Variable},
};

use crate::{
    quantization::{quantized_float::QuantizedFloat, quantizer::Quantizer},
    tensor::{shape::Shape, tensor::Tensor},
};

#[derive(Debug, Clone)]
pub(crate) struct ReluOp {
    pub(crate) id: usize,
    pub(crate) input_id: usize,
}

impl ReluOp {
    fn new(id: usize, input_id: usize) -> Self {
        Self { id, input_id }
    }

    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<QuantizedFloat>>,
    ) -> Tensor<QuantizedFloat> {
        let input = history.get(&self.input_id).unwrap();

        let params = ReluParams::new(input.data.len(), input.shape.clone(), input.data.len());

        params.create_circuit(api, input.clone())
    }
}

#[derive(Debug, Clone, Default)]
struct ReluParams {
    input_len: usize,
    input_shape: Shape,
    target_len: usize,
}

impl ReluParams {
    fn new(input_len: usize, input_shape: Shape, target_len: usize) -> Self {
        Self {
            input_len,
            input_shape,
            target_len,
        }
    }
    fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        input: Tensor<QuantizedFloat>,
    ) -> Tensor<QuantizedFloat> {
        let query: Vec<Variable> = input.data.iter().map(|val| val.to_var()).collect();

        let hint = api.new_hint("relu_hint", &query, self.target_len);

        let res = hint.iter().map(|val| QuantizedFloat::new(*val)).collect();

        Tensor::new(Some(res), input.shape.clone())
    }
}

pub fn relu_hint(input: &[BN254], output: &mut [BN254]) -> Result<(), Error> {
    // TODO: look for a way of getting the quantizer in the hint
    let quantizer = Quantizer::<16> {};

    for i in 0..input.len() {
        let input = quantizer.dequantize(&input[i]);
        let zero = quantizer.dequantize(&BN254::zero());

        if input < zero {
            output[i] = BN254::zero();
        } else {
            output[i] = quantizer.quantize(input);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::{
        ir::op::relu::{relu_hint, ReluParams},
        quantization::{quantized_float::QuantizedFloat, quantizer::Quantizer},
        tensor::{shape::Shape, tensor::Tensor},
    };
    use expander_compiler::{
        compile::CompileOptions,
        declare_circuit,
        field::BN254,
        frontend::{compile, BN254Config, Define, HintRegistry, RootAPI, Variable},
    };

    #[test]
    fn test_relu() {
        let mut hint_registery = HintRegistry::<BN254>::new();
        hint_registery.register("relu_hint", relu_hint);

        declare_circuit!(ReluCircuit {
            input: [Variable],
            target: [Variable],
            params: ReluParams
        });

        impl ReluCircuit<Variable> {
            fn new(params: ReluParams) -> Self {
                let mut circuit = Self::default();

                circuit
                    .target
                    .resize(params.target_len, Variable::default());
                circuit.input.resize(params.input_len, Variable::default());

                circuit.params = params;
                circuit
            }
        }

        impl Define<BN254Config> for ReluCircuit<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let input = self
                    .input
                    .iter()
                    .map(|val| QuantizedFloat::new(*val))
                    .collect();

                let res = self.params.create_circuit(
                    api,
                    Tensor::new(Some(input), self.params.input_shape.clone()),
                );

                for i in 0..res.data.len() {
                    api.assert_is_equal(res.data[i].to_var(), self.target[i]);
                }
            }
        }

        let quantizer = Quantizer::<16> {};

        let input_data = vec![2., 0., -1., -5., 7., -2.]
            .into_iter()
            .map(|val| Quantizer::quantize(&quantizer, val))
            .collect();

        let input = Tensor::new(Some(input_data), Shape::new(vec![1, 6]));

        let target: Vec<BN254> = vec![2., 0., 0., 0., 7., 0.]
            .into_iter()
            .map(|val| Quantizer::quantize(&quantizer, val))
            .collect();

        let params = ReluParams {
            input_len: input.data.len(),
            input_shape: input.shape,
            target_len: target.len(),
        };

        let compiled_circuit: expander_compiler::frontend::CompileResult<BN254Config> =
            compile(&ReluCircuit::new(params.clone()), CompileOptions::default()).unwrap();

        let assignment = ReluCircuit {
            target,
            params,
            input: input.data,
        };

        let witness = compiled_circuit
            .witness_solver
            .solve_witness_with_hints(&assignment, &mut hint_registery)
            .unwrap();

        let output = compiled_circuit.layered_circuit.run(&witness);

        output.iter().for_each(|val| assert!(val));

        dbg!(&output);
    }
}
