use expander_compiler::field::{FieldArith, BN254};
use expander_compiler::frontend::extra::debug_eval;
use expander_compiler::frontend::{BN254Config, EmptyHintCaller, Variable};
use tract_core::internal::tract_itertools::Itertools;
use zkml::cmd::compile_circuit;
use zkml::model_circuit::_ModelCircuit;
use zkml::quantization::quantizer::Quantizer;

// this model tries to estimate 2 * norm_x + 46

const X_MEAN: f32 = 49.5;
const X_STD: f32 = 28.86607004772212;
const Y_MEAN: f32 = 145.0;
const Y_STD: f32 = 57.73214009544424;

fn normalize(x: f32) -> f32 {
    (x - X_MEAN) / X_STD
}

fn denormalize(y: f32) -> f32 {
    y * Y_STD + Y_MEAN
}

fn hex_to_bn254(hex_str: &str) -> BN254 {
    let bytes: [u8; 32] = hex::decode(hex_str)
        .unwrap()
        .iter()
        .cloned()
        .rev()
        .collect_vec()
        .try_into()
        .unwrap();
    BN254::from_bytes(&bytes).unwrap()
}

fn main() {
    // run model with x = [90], expected_result ~= 226
    let x = normalize(90.0);

    let quantizer = Quantizer::<16> {};
    let input = vec![quantizer.quantize(x)];

    let output_hex = "00000000000000000000000000000000000000000000000000000000000166d6";
    let output = vec![hex_to_bn254(output_hex)];
    let result = output
        .iter()
        .map(|v| denormalize(quantizer.dequantize(v)))
        .collect_vec();

    let build_result = compile_circuit(
        "../models/linear_regression.onnx".into(),
        input,
        output,
        &quantizer,
    );
    println!("output: {}", result[0]);

    let witness = build_result
        .compile_result
        .witness_solver
        .solve_witness(&build_result.assignment)
        .unwrap();
    let run_result = build_result.compile_result.layered_circuit.run(&witness);
    assert!(run_result.iter().all(|v| *v));

    // debug_eval::<BN254Config, _ModelCircuit<Variable>, _ModelCircuit<BN254>, EmptyHintCaller>(
    //     &build_result.model,
    //     &build_result.assignment,
    //     EmptyHintCaller::new(),
    // );
}
