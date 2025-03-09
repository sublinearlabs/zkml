use std::collections::HashMap;

use crate::ir::op::NodeOp;
use crate::quantization::quantized_float::QuantizedFloat;
use expander_compiler::{
    declare_circuit,
    field::BN254,
    frontend::{Config, Define, Variable},
};

use crate::tensor::tensor::Tensor;

#[derive(Debug, Clone)]
pub struct ModelParameters {
    pub input_len: usize,
    pub output_len: usize,
    pub weight_len: usize,
    pub ops: Vec<NodeOp>,
}

pub struct AssignmentParameters {
    pub inputs: Vec<BN254>,
    pub weights: Vec<BN254>,
    pub outputs: Vec<BN254>,
    pub shift: BN254,
}

declare_circuit!(_ModelCircuit {
    input: [Variable],
    output: [Variable],
    weights: [Variable],
    shift: Variable,
    ops: [NodeOp],
});

pub type ModelCircuit = _ModelCircuit<Variable>;

impl ModelCircuit {
    pub(crate) fn new_circuit(params: &ModelParameters) -> Self {
        let mut new_circuit = Self::default();

        new_circuit
            .input
            .resize(params.input_len, Variable::default());
        new_circuit
            .output
            .resize(params.output_len, Variable::default());
        new_circuit
            .weights
            .resize(params.weight_len, Variable::default());
        new_circuit.ops.resize(params.ops.len(), NodeOp::Unknown);

        for i in 0..params.ops.len() {
            new_circuit.ops[i] = params.ops[i].clone();
        }

        new_circuit
    }

    pub fn new_assignment(params: &AssignmentParameters) -> _ModelCircuit<BN254> {
        let mut new_assignment = _ModelCircuit::<BN254>::default();

        new_assignment
            .input
            .resize(params.inputs.len(), BN254::default());
        new_assignment
            .output
            .resize(params.outputs.len(), BN254::default());
        new_assignment
            .weights
            .resize(params.weights.len(), BN254::default());
        new_assignment.shift = params.shift;

        for i in 0..params.weights.len() {
            new_assignment.weights[i] = params.weights[i];
        }
        for i in 0..params.inputs.len() {
            new_assignment.input[i] = params.inputs[i];
        }
        for i in 0..params.outputs.len() {
            new_assignment.output[i] = params.outputs[i];
        }

        new_assignment
    }
}

impl<C: Config> Define<C> for ModelCircuit {
    fn define<Builder: expander_compiler::frontend::RootAPI<C>>(&self, api: &mut Builder) {
        let mut history: HashMap<usize, Tensor<QuantizedFloat>> = HashMap::new();

        for op in &self.ops {
            let circuit_eval_result =
                op.create_circuit(api, &history, &self.input, &self.weights, self.shift);
            history.insert(op.id(), circuit_eval_result);
        }

        let last_circuit = self.ops.last().unwrap().id();

        let expected = history.get(&last_circuit).unwrap();

        for i in 0..self.output.len() {
            api.display(
                format!("out-{}", i).as_str(),
                expected.data[i].clone().to_var(),
            );
            api.assert_is_equal(expected.data[i].to_var(), self.output[i]);
        }
    }
}

#[cfg(test)]
mod tests {

    use expander_compiler::field::BN254;
    use expander_compiler::frontend::BN254Config;
    use expander_compiler::{
        compile::CompileOptions,
        frontend::{compile, CompileResult},
    };

    use crate::ir::op::add::AddOp;
    use crate::ir::op::tensor_view::{TensorViewOp, ViewType};
    use crate::ir::op::NodeOp;
    use crate::tensor::shape::Shape;

    use super::{AssignmentParameters, ModelCircuit, ModelParameters};

    #[test]
    fn test_model_circuit() {
        let model_params = ModelParameters {
            input_len: 1,
            output_len: 1,
            weight_len: 1,
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

        let assignment_params = AssignmentParameters {
            weights: vec![BN254::from(3_u64)],
            inputs: vec![BN254::from(5_u64)],
            outputs: vec![BN254::from(10_u64)],
            // scale not necessary for this example
            shift: BN254::one(),
        };

        let compiled_result: CompileResult<BN254Config> = compile(
            &ModelCircuit::new_circuit(&model_params),
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = ModelCircuit::new_assignment(&assignment_params);

        let witness = compiled_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compiled_result.layered_circuit.run(&witness);

        output.iter().for_each(|val| assert!(val));
    }
}
