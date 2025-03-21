use std::collections::HashMap;

use expander_compiler::{
    field::BN254,
    frontend::{Config, Error, RootAPI, Variable},
};

use crate::{quantization::quantizer::Quantizer, tensor::tensor::Tensor};

// struct Relu {
//     id: usize,
//     input_id: usize,
// }

// impl Relu {
//     fn new(id: usize, input_id: usize) -> Self {
//         Self { id, input_id }
//     }
// }

#[derive(Debug, Clone, Default)]
struct ReluParams {
    input_id: usize,
    input: Tensor<BN254>,
    target_len: usize,
}

impl ReluParams {
    fn compute<T: Default + Clone + PartialOrd>(&self, input: Tensor<T>) -> Tensor<T> {
        let mut res = vec![];
        for val in input.data {
            if val < T::default() {
                res.push(T::default());
            } else {
                res.push(val);
            }
        }
        Tensor::new(Some(res), input.shape)
    }

    fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
        input_id: &usize,
    ) -> Tensor<Variable> {
        let query = history.get(input_id).unwrap();

        let hint = api.new_hint("relu_hint", &query.data, self.target_len);

        Tensor::new(Some(hint), query.shape.clone())
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
    use std::collections::HashMap;

    use crate::{
        ir::op::relu::{relu_hint, ReluParams},
        quantization::quantizer::Quantizer,
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
                circuit
                    .input
                    .resize(params.input.data.len(), Variable::default());

                circuit.params = params;
                circuit
            }
        }

        impl Define<BN254Config> for ReluCircuit<Variable> {
            fn define<Builder: RootAPI<BN254Config>>(&self, api: &mut Builder) {
                let mut history = HashMap::new();

                let input_id = self.params.input_id;

                let input_data =
                    Tensor::new(Some(self.input.clone()), self.params.input.shape.clone());

                history.insert(input_id, input_data);

                let res = self.params.create_circuit(api, &history, &input_id);

                for i in 0..res.data.len() {
                    api.assert_is_equal(res.data[i], self.target[i]);
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
            input: input.clone(),
            target_len: target.len(),
            input_id: 0,
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

        dbg!(&output);
    }
}
