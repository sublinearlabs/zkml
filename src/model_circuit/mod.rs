use expander_compiler::{
    declare_circuit,
    field::M31,
    frontend::{Config, Define, Variable},
};

#[derive(Debug, Clone)]
struct ModelParameters {
    input_len: usize,
    output_len: usize,
    weights: Vec<M31>,
    input: Vec<M31>,
    output: Vec<M31>,
}

declare_circuit!(_ModelCircuit {
    input: [Variable],
    output: [Variable],
    weights: [Variable],
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
            .resize(params.weights.len(), Variable::default());

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
            .resize(params.weights.len(), M31::default());

        for i in 0..params.weights.len() {
            new_assignment.weights[i] = params.weights[i];
        }
        for i in 0..params.input.len() {
            new_assignment.input[i] = params.input[i];
        }
        for i in 0..params.output.len() {
            new_assignment.output[i] = params.output[i];
        }

        new_assignment
    }
}

impl<C: Config> Define<C> for ModelCircuit {
    fn define<Builder: expander_compiler::frontend::RootAPI<C>>(&self, api: &mut Builder) {
        let c = api.add(self.input[0], self.weights[0]);
        api.assert_is_equal(c, self.output[0]);
    }
}

#[cfg(test)]
mod tests {

    use expander_compiler::{
        compile::CompileOptions,
        field::M31,
        frontend::{compile, CompileResult, M31Config},
    };

    use super::{ModelCircuit, ModelParameters};

    #[test]
    fn test_model_circuit() {
        let params = ModelParameters {
            input_len: 1,
            output_len: 1,
            weights: vec![M31::from(3)],
            input: vec![M31::from(5)],
            output: vec![M31::from(8)],
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
