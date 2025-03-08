use std::collections::HashMap;

use crate::ir::op::NodeOp;
use expander_compiler::{
    declare_circuit,
    field::BN254,
    frontend::{Config, Define, Variable},
};

use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
pub(crate) struct ModelParameters {
    pub(crate) input_len: usize,
    pub(crate) output_len: usize,

    pub(crate) weights: Vec<BN254>,
    pub(crate) ops: Vec<NodeOp>,

    pub(crate) input: Vec<BN254>,
    pub(crate) output: Vec<BN254>,
}

declare_circuit!(_ModelCircuit {
    input: [Variable],
    output: [Variable],
    weights: [Variable],
    ops: [NodeOp],
});

type ModelCircuit = _ModelCircuit<Variable>;

impl ModelCircuit {
    type Params = ModelParameters;

    type Assignment = _ModelCircuit<BN254>;

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
            .resize(params.weights.len(), Variable::default());
        new_circuit.ops.resize(params.ops.len(), NodeOp::Unknown);

        for i in 0..params.ops.len() {
            new_circuit.ops[i] = params.ops[i].clone();
        }

        new_circuit
    }

    fn new_assignment(params: &Self::Params) -> Self::Assignment {
        let mut new_assignment = Self::Assignment::default();

        new_assignment
            .input
            .resize(params.input_len, BN254::default());
        new_assignment
            .output
            .resize(params.output_len, BN254::default());
        new_assignment
            .weights
            .resize(params.weights.len(), BN254::default());
        new_assignment.ops.resize(params.ops.len(), NodeOp::Unknown);

        for i in 0..params.weights.len() {
            new_assignment.weights[i] = params.weights[i];
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
            let circuit_eval_result = op.create_circuit(api, &history, &self.input, &self.weights);
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

    use expander_compiler::field::BN254;
    use expander_compiler::frontend::BN254Config;
    use expander_compiler::{
        compile::CompileOptions,
        frontend::{compile, CompileResult, M31Config},
    };

    use crate::ir::op::add::AddOp;
    use crate::ir::op::tensor_view::{TensorViewOp, ViewType};
    use crate::ir::op::NodeOp;
    use crate::tensor::{shape::Shape, tensor::Tensor};

    use super::{ModelCircuit, ModelParameters};

    #[test]
    fn test_model_circuit() {
        let params = ModelParameters {
            input_len: 2,
            output_len: 1,
            weights: vec![BN254::from(3_u64)],
            input: vec![BN254::from(5_u64)],
            output: vec![BN254::from(10_u64)],
            ops: vec![
                NodeOp::TensorView(TensorViewOp {
                    id: 0,
                    tensor_type: ViewType::Input,
                    start_index: 0,
                    shape: Shape::new(vec![1, 1]),
                }),
                NodeOp::TensorView(TensorViewOp {
                    id: 1,
                    tensor_type: ViewType::Weights,
                    start_index: 0,
                    shape: Shape::new(vec![1, 1]),
                }),
                NodeOp::Add(AddOp {
                    id: 5,
                    lhs_id: 0,
                    rhs_id: 0,
                }),
            ],
        };

        let compiled_result: CompileResult<BN254Config> = compile(
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
