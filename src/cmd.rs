use crate::ir::load_onnx::{load_onnx, model_graph_to_ir};
use crate::model_circuit::{ModelCircuit, ModelParameters, _ModelCircuit};
use expander_compiler::field::{FieldArith, BN254};
use expander_compiler::frontend::{compile, BN254Config, CompileOptions, CompileResult};
use std::path::PathBuf;

fn compile_circuit(path: PathBuf) -> CompileResult<BN254Config> {
    const FRACTIONAL_BITS: u8 = 15;
    let tract_graph = load_onnx(path);
    let ir = model_graph_to_ir(&tract_graph);

    let scale: u64 = 1 << FRACTIONAL_BITS;

    let model_params = ModelParameters {
        input_len: ir.input_count,
        output_len: ir.output_ids.len(),
        scale_inv: BN254::from(scale).inv().unwrap(),
        weights: ir.quantize_constants::<FRACTIONAL_BITS>(),
        ops: ir.ops,
        input: vec![],
        output: vec![],
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
