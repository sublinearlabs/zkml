use std::collections::HashMap;

use expander_compiler::{
    field::{FieldArith, M31},
    frontend::{Config, RootAPI, Variable},
};

use crate::tensor::{shape::Shape, tensor::Tensor};

// struct Relu {
//     id: usize,
//     name: String,
//     input_id: usize,
// }

// impl Relu {
//     fn new(id: usize, name: String, input_id: usize) -> Self {
//         Self { id, name, input_id }
//     }
// }

#[derive(Debug, Clone, Default)]
struct ReluParams {
    input_id: usize,
    input: Tensor<M31>,
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
        weight_value: &Vec<Variable>,
        lookup_table: &Vec<Variable>,
    ) -> Tensor<Variable> {
        let mut res_data = vec![];

        let input_data = history.get(input_id).unwrap();

        for input in &input_data.data {
            // let c = api.unconstrained_greater(C::CircuitField::zero(), input);
            let c = api.unconstrained_greater(C::CircuitField::zero(), input);
            res_data.push(c);
        }

        Tensor::new(Some(res_data), input_data.shape.clone())
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::ops::Neg;

    use crate::{
        ir::op::relu::ReluParams,
        tensor::{shape::Shape, tensor::Tensor},
    };
    use expander_compiler::{
        builder::basic::Builder,
        compile::CompileOptions,
        declare_circuit,
        field::{FieldArith, FieldModulus, M31},
        frontend::{compile, BasicAPI, Config, Define, M31Config, RootAPI, Variable},
    };

    #[test]
    fn test_relu() {
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

        impl Define<M31Config> for ReluCircuit<Variable> {
            fn define<Builder: RootAPI<M31Config>>(&self, api: &mut Builder) {
                let mut history = HashMap::new();

                let input_id = self.params.input_id;

                let input_data =
                    Tensor::new(Some(self.input.clone()), self.params.input.shape.clone());

                history.insert(input_id, input_data);

                let res = self
                    .params
                    .create_circuit(api, &history, &input_id, &vec![], &vec![]);

                for i in 0..res.data.len() {
                    api.assert_is_equal(res.data[i], self.target[i]);
                }
            }
        }

        // let input: Tensor<M31> = Tensor::new(Some(vec![M31::from(9)]), Shape::new(vec![1]));
        // let target: Vec<M31> = vec![M31::from(10)];
        // let input: Tensor<M31> = Tensor::new(Some(vec![M31::from(3),M31::from(5),M31::from(1).neg(),M31::zero()]), Shape::new(vec![1, 4]));

        let input = Tensor::new(
            Some(vec![
                M31::from(2),
                M31::from(0),
                M31::from(1).neg(),
                M31::from(5).neg(),
                M31::from(7),
                M31::from(2).neg(),
            ]),
            Shape::new(vec![1, 6]),
        );

        let target: Vec<M31> = vec![
            M31::from(2),
            M31::zero(),
            M31::zero(),
            M31::zero(),
            M31::from(7),
            M31::zero(),
        ];

        let params = ReluParams {
            input: input.clone(),
            target_len: target.len(),
            input_id: 0,
        };

        let compiled_circuit: expander_compiler::frontend::CompileResult<M31Config> =
            compile(&ReluCircuit::new(params.clone()), CompileOptions::default()).unwrap();

        let assignment = ReluCircuit {
            target,
            params,
            input: input.data,
        };

        let witness = compiled_circuit
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compiled_circuit.layered_circuit.run(&witness);

        dbg!(&output);
    }
}
