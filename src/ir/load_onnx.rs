use std::path::PathBuf;
use tract_core::internal::tract_itertools::Itertools;

use crate::ir::intermediate_representation::IR;
use tract_onnx::prelude::*;

pub(crate) fn load_onnx(path: PathBuf) -> Graph<TypedFact, Box<dyn TypedOp>> {
    onnx()
        .model_for_path(path)
        .unwrap()
        // .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1)))
        // .unwrap()
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap()
}

pub(crate) fn model_graph_to_ir(model_graph: &Graph<TypedFact, Box<dyn TypedOp>>) -> IR {
    let mut input_count = 0;
    let mut constants = vec![];
    let output_ids = model_graph
        .outputs
        .iter()
        .map(|outlet| {
            // we do not handle output outlets with slot not equal to 0 for now
            // adding an assert for early catch
            assert_eq!(outlet.slot, 0, "encountered model with outlet slot != 0");
            outlet.node
        })
        .collect_vec();

    let mut ops = vec![];

    IR::new(input_count, constants, output_ids, ops)
}

#[cfg(test)]
mod tests {
    use super::{load_onnx, model_graph_to_ir};
    #[test]
    fn test_load_onnx() {
        let model_graph = load_onnx("models/linear_regression.onnx".into());
        dbg!(&model_graph);
        let _ = model_graph_to_ir(&model_graph);
    }
}
