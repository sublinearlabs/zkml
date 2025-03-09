use expander_compiler::field::{FieldArith, BN254};
use zkml::cmd::compile_circuit;
use zkml::model_circuit::{AssignmentParameters, ModelParameters, _ModelCircuit};
use zkml::quantization::quantizer::Quantizer;

fn main() {
    let quantizer = Quantizer::<15> {};
    let compile_result = compile_circuit("models/linear_regression.onnx".into(), &quantizer);
    let input = quantizer.quantize(4.5_f32);
    let assignment_parameters = AssignmentParameters {
        inputs: vec![input],
        outputs: vec![input],
        weights: vec![input],
        scale_inv: BN254::from(quantizer.scale()).inv().unwrap(),
    };
    let assignment = _ModelCircuit::new_assignment(&assignment_parameters);
    let witness = compile_result
        .witness_solver
        .solve_witness(&assignment)
        .unwrap();
    let a = compile_result.layered_circuit.run(&witness);
    dbg!(&a);
}
