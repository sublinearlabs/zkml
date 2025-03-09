use expander_compiler::field::{FieldArith, BN254};
use zkml::cmd::compile_circuit;
use zkml::model_circuit::{AssignmentParameters, ModelParameters, _ModelCircuit};
use zkml::quantization::quantizer::Quantizer;

fn main() {
    let quantizer = Quantizer::<15> {};
    let input = vec![quantizer.quantize(4.5_f32)];
    let build_result = compile_circuit(
        "models/linear_regression.onnx".into(),
        input.clone(),
        input,
        &quantizer,
    );
    let witness = build_result.compile_result.witness_solver.solve_witness(&build_result.assignment).unwrap();
    let run_result = build_result.compile_result.layered_circuit.run(&witness);
    dbg!(&run_result);
}
