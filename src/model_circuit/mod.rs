use std::collections::HashMap;

use circuit_std_rs::StdCircuit;
use expander_compiler::{
    circuit::ir::common::{Circuit, IrConfig},
    declare_circuit,
    frontend::{Config, Define, RootAPI, Variable},
};
use rand::RngCore;

use crate::supported_ops::SupportedOps;

#[derive(Debug, Clone)]
struct ModelParameters {
    // Total number of Op in the model
    total_ops: usize,
    // Flattened input length
    input_len: usize,
    // Flattened output len
    output_len: usize,
    // Flattened weight len
    weight_len: usize,
}

declare_circuit!(_ModelCircuit {
    // Flattened input to the circuit
    input: [Variable],
    // Flattened output of the circuit
    output: [Variable],
    // Ops Ir info
    ops: [SupportedOps],
    // Flattened model weight
    weight: [Variable]
});

type ModelCircuit = _ModelCircuit<Variable>;

impl<C: Config> StdCircuit<C> for ModelCircuit {
    type Params = ModelParameters;

    type Assignment = _ModelCircuit<C::CircuitField>;

    fn new_circuit(params: &Self::Params) -> Self {
        let mut circuit = Self::default();

        circuit.input.resize(params.input_len, Variable::default());
        circuit
            .output
            .resize(params.output_len, Variable::default());
        circuit.ops.resize(params.total_ops, SupportedOps::Unknown);
        circuit
            .weight
            .resize(params.weight_len, Variable::default());

        circuit
    }

    fn new_assignment(params: &Self::Params, rng: impl RngCore) -> Self::Assignment {
        let mut assignment = Self::Assignment::default();

        assignment
            .input
            .resize(params.input_len, C::CircuitField::default());
        assignment
            .output
            .resize(params.output_len, C::CircuitField::default());
        assignment
            .ops
            .resize(params.total_ops, SupportedOps::Unknown);
        assignment
            .weight
            .resize(params.weight_len, C::CircuitField::default());

        assignment
    }
}

impl<C: Config> Define<C> for ModelCircuit {
    fn define<Builder: expander_compiler::frontend::RootAPI<C>>(&self, api: &mut Builder) {
        let mut history: HashMap<usize, Vec<Variable>> = HashMap::new();

        for op in &self.ops {
            let circuit = generate_circuit(op, api);
            // circuit.
        }
    }
}

fn generate_circuit<C: Config, Irc: IrConfig, Builder: RootAPI<C>>(
    op: &SupportedOps,
    api: Builder,
) -> Circuit<Irc> {
    todo!()
}

#[cfg(test)]
mod tests {
    use circuit_std_rs::StdCircuit;
    use expander_compiler::{compile::CompileOptions, field::M31, frontend::compile};

    use crate::supported_ops::{
        load_onnx::{load_onnx, model_graph_to_ir},
        SupportedAdd, SupportedOps,
    };

    use super::{ModelCircuit, ModelParameters};

    fn test_create_model_circuit() {
        let model_graph = load_onnx("models/test_onnx_model.onnx".into());
        let ir_info = model_graph_to_ir(&model_graph);

        let params = ModelParameters {
            output_len: 1,
            input_len: 2,
            total_ops: 1,
            weight_len: 1,
        };

        let compiled_result = compile(
            &ModelCircuit::new_circuit(&params),
            CompileOptions::default(),
        )
        .unwrap();

        let assignment = ModelCircuit {
            input: vec![M31::from(3), M31::from(3)],
            output: vec![M31::from(12)],
            ops: vec![SupportedOps::Add(SupportedAdd {
                id: 1,
                name: "Add".to_string(),
            })],
            weight: vec![M31::from(2)],
        };

        let witness = compiled_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();

        let output = compiled_result.layered_circuit.run(&witness);

        output.iter().for_each(|val| assert!(val));
    }
}
