use crate::ir::load_onnx::{load_onnx, model_graph_to_ir};
use crate::model_circuit::{AssignmentParameters, ModelCircuit, ModelParameters, _ModelCircuit};
use crate::quantization::quantizer::Quantizer;
use expander_compiler::field::{FieldArith, BN254};
use expander_compiler::frontend::{compile, BN254Config, CompileOptions, CompileResult};
use std::path::PathBuf;

pub struct CompilationResult {
    pub compile_result: CompileResult<BN254Config>,
    pub assignment: _ModelCircuit<BN254>,
}

pub fn compile_circuit<const N: u8>(
    path: PathBuf,
    inputs: Vec<BN254>,
    outputs: Vec<BN254>,
    quantizer: &Quantizer<N>,
) -> CompilationResult {
    let tract_graph = load_onnx(path);
    let ir = model_graph_to_ir(&tract_graph);

    let model_params = ModelParameters {
        input_len: ir.input_count,
        output_len: ir.output_ids.len(),
        weight_len: ir.constants.len(),
        ops: ir.ops.clone(),
    };

    let compile_result = compile(
        &ModelCircuit::new_circuit(&model_params),
        CompileOptions::default(),
    )
    .expect("failed to compile circuit");

    let weights = ir.quantize_constants(&quantizer);

    let assignment_parameters = AssignmentParameters {
        inputs,
        outputs,
        weights,
        scale_inv: BN254::from(quantizer.scale()).inv().unwrap(),
    };

    let assignment = _ModelCircuit::new_assignment(&assignment_parameters);

    CompilationResult {
        compile_result,
        assignment,
    }
}
