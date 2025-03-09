use crate::ir::load_onnx::{load_onnx, model_graph_to_ir};
use crate::model_circuit::{ModelCircuit, ModelParameters, _ModelCircuit};
use crate::quantization::quantizer::Quantizer;
use expander_compiler::field::{FieldArith, BN254};
use expander_compiler::frontend::{compile, BN254Config, CompileOptions, CompileResult};
use std::path::PathBuf;

pub fn compile_circuit<const N: u8>(path: PathBuf) -> CompileResult<BN254Config> {
    let tract_graph = load_onnx(path);
    let ir = model_graph_to_ir(&tract_graph);

    let model_params = ModelParameters {
        input_len: ir.input_count,
        output_len: ir.output_ids.len(),
        weight_len: ir.constants.len(),
        ops: ir.ops,
    };

    compile(
        &ModelCircuit::new_circuit(&model_params),
        CompileOptions::default(),
    )
    .expect("failed to compile circuit")
}

#[cfg(test)]
mod test {
    use crate::cmd::compile_circuit;

    #[test]
    fn test_compile_circuit() {
        dbg!(compile_circuit("models/linear_regression.onnx".into()).layered_circuit);
    }
}
