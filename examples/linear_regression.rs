use expander_compiler::field::{BN254, FieldArith};
use zkml::cmd::compile_circuit;
use zkml::model_circuit::{_ModelCircuit, ModelParameters};
use zkml::quantization::quantizer::Quantizer;
use zkml::tensor::tensor::Tensor;

fn main() {
    let quantizer = Quantizer::<15>{};
    let compile_result = compile_circuit("models/linear_regression.onnx".into());
    let input = quantizer.quantize(4.5_f32);
    let model_params = ModelParameters {
        input_len: 1,
        output_len: 1,
        weights: vec![input, input],
        ops: vec![],
        input: vec![input],
        output: vec![input],
        scale_inv: BN254::from((1 << 15) as u64).inv().unwrap()
    };

    let assignment = _ModelCircuit::new_assignment(&model_params);
    let witness = compile_result.witness_solver.solve_witness(&assignment).unwrap();
    let a = compile_result.layered_circuit.run(&witness);
    dbg!(&a);
}