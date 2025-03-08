use std::collections::HashMap;

use crate::ir::ops::Ops;
use expander_compiler::{
    declare_circuit,
    field::M31,
    frontend::{Config, Define, Variable},
};

use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
struct ModelParameters {
    input_len: usize,
    output_len: usize,

    weights: Vec<Tensor<M31>>,
    ops: Vec<Ops>,

    input: Vec<M31>,
    output: Vec<M31>,
}

declare_circuit!(_ModelCircuit {
    input: [Variable],
    output: [Variable],
    weights: [[Variable]],
    ops: [Ops],
});

type ModelCircuit = _ModelCircuit<Variable>;

impl ModelCircuit {
    type Params = ModelParameters;

    type Assignment = _ModelCircuit<M31>;

    fn new_circuit(params: &Self::Params) -> Self {
        let mut new_circuit = Self::default();

        new_circuit
            .input
            .resize(params.input_len, Variable::default());
        new_circuit
            .output
            .resize(params.output_len, Variable::default());
        new_circuit
            .weights
            .resize(params.weights.len(), vec![Variable::default()]);
        new_circuit.ops.resize(params.ops.len(), Ops::Unknown);

        for i in 0..params.ops.len() {
            new_circuit.ops[i] = params.ops[i].clone();
        }

        new_circuit
    }

    fn new_assignment(params: &Self::Params) -> Self::Assignment {
        let mut new_assignment = Self::Assignment::default();

        new_assignment
            .input
            .resize(params.input_len, M31::default());
        new_assignment
            .output
            .resize(params.output_len, M31::default());
        new_assignment
            .weights
            .resize(params.weights.len(), vec![M31::default()]);
        new_assignment.ops.resize(params.ops.len(), Ops::Unknown);

        for i in 0..params.weights.len() {
            new_assignment.weights[i] = params.weights[i].data.clone();
        }
        for i in 0..params.input.len() {
            new_assignment.input[i] = params.input[i];
        }
        for i in 0..params.output.len() {
            new_assignment.output[i] = params.output[i];
        }
        for i in 0..params.ops.len() {
            new_assignment.ops[i] = params.ops[i].clone();
        }

        new_assignment
    }
}

impl<C: Config> Define<C> for ModelCircuit {
    fn define<Builder: expander_compiler::frontend::RootAPI<C>>(&self, api: &mut Builder) {
        let mut history: HashMap<usize, Tensor<Variable>> = HashMap::new();

        for op in &self.ops {
            let circuit_eval_result = op.create_circuit(api, &history, &self.input);
            history.insert(op.id(), circuit_eval_result);
        }

        let last_circuit = self.ops.last().unwrap().id();

        let expected = history.get(&last_circuit).unwrap();

        dbg!(&history);
        dbg!(&expected);
        dbg!(&self.output);

        // TODO: Handle multiple outputs
        for i in 0..self.output.len() {
            api.assert_is_equal(expected.data[i], self.output[i]);
        }
    }
}

#[cfg(test)]
mod tests {

    use expander_compiler::{
        compile::CompileOptions,
        field::M31,
        frontend::{compile, CompileResult, M31Config},
    };

    use crate::ir::ops::add::AddOp;
    use crate::ir::ops::tensor_view::{TensorViewOp, ViewType};
    use crate::ir::ops::Ops;
    use crate::tensor::{shape::Shape, tensor::Tensor};

    use super::{ModelCircuit, ModelParameters};

    #[test]
    fn test_model_circuit() {
        let params = ModelParameters {
            input_len: 2,
            output_len: 1,
            weights: vec![Tensor::new(
                Some(vec![M31::from(3)]),
                Shape::new(vec![1, 1]),
            )],
            input: vec![M31::from(5)],
            output: vec![M31::from(10)],
            ops: vec![
                Ops::TensorView(TensorViewOp {
                    id: 0,
                    tensor_type: ViewType::Input,
                    start_index: 0,
                    shape: Shape::new(vec![1, 1]),
                }),
                Ops::Add(AddOp {
                    id: 5,
                    lhs_id: 0,
                    rhs_id: 0,
                }),
            ],
        };

        let compiled_result: CompileResult<M31Config> = compile(
            &ModelCircuit::new_circuit(&params),
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = ModelCircuit::new_assignment(&params);

        let witness = compiled_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compiled_result.layered_circuit.run(&witness);

        output.iter().for_each(|val| assert!(val));
    }
}
