use expander_compiler::field::{FieldArith, BN254};
use expander_compiler::frontend::extra::debug_eval;
use expander_compiler::frontend::{BN254Config, EmptyHintCaller, Variable};
use tract_core::internal::tract_itertools::Itertools;
use tract_core::num_traits::ToPrimitive;
use zkml::cmd::compile_circuit;
use zkml::model_circuit::_ModelCircuit;
use zkml::quantization::quantizer::Quantizer;

// This model is trained on the california housing dataset from sklearn

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
    // Model input:
    // x = [-1.1551, -0.2863, -0.5207, -0.1717, -0.0303,  0.0674,  0.1951,  0.2853],
    // Expected_output
    // expected_out ~= [-0.0457]

    let x = vec![
        -1.1551, -0.2863, -0.5207, -0.1717, -0.0303, 0.0674, 0.1951, 0.2853,
    ];

    let quantizer = Quantizer::<16> {};
    let input = x.iter().map(|val| quantizer.quantize(*val)).collect();

    // TODO: output hex
    let output_hex = "22769f827e728891563c62f5ced0eebf300e9f90057859b3e4391f15e07054b1";
    let output = vec![hex_to_bn254(output_hex)];

    let result = output.iter().map(|v| quantizer.dequantize(v)).collect_vec();

    dbg!(&result);

    let build_result = compile_circuit(
        "../models/linear_regression2.onnx".into(),
        input,
        output,
        &quantizer,
    );

    dbg!(&build_result.assignment.input);

    // let witness = build_result.compile_result.witness_solver.solve_witness(&build_result.assignment).unwrap();
    // let run_result = build_result.compile_result.layered_circuit.run(&witness);
    // dbg!(&run_result);

    debug_eval::<BN254Config, _ModelCircuit<Variable>, _ModelCircuit<BN254>, EmptyHintCaller>(
        &build_result.model,
        &build_result.assignment,
        EmptyHintCaller::new(),
    );
}

// cargo run --package zkml --example linear_regression2
// -1.1551 * -0.2863 = 0.33070513
