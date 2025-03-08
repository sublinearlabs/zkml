use std::fmt::Debug;
use std::path::PathBuf;
use tract_core::internal::tract_itertools::Itertools;

use crate::ir::intermediate_representation::IR;
use crate::ir::ops::tensor_view::{TensorViewOp, ViewType};
use crate::ir::ops::Ops;
use crate::ir::TractOps;
use crate::tensor::shape::Shape;
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

    for node in &model_graph.nodes {
        let op = match node.op.name().as_ref() {
            "Source" => parse_source(node.into(), &mut input_count),
            unknown_op => panic!("unsupported node: {}", unknown_op),
        };
        dbg!(&op);
        ops.push(op);
    }

    IR::new(input_count, constants, output_ids, ops)
}

fn parse_source<F: Fact, O: Debug>(node: &Node<F, O>, input_index: &mut usize) -> Ops {
    assert_eq!(node.outputs.len(), 1);
    let shape_data = &node.outputs[0]
        .fact
        .to_typed_fact()
        .expect("should be typed model")
        .shape;
    let shape_dims = shape_data
        .as_concrete()
        .expect("failed to get concrete shape");
    let shape = Shape::new(shape_dims.to_vec());
    let volume = shape.volume();
    let op = Ops::TensorView(TensorViewOp {
        id: node.id,
        tensor_type: ViewType::Input,
        start_index: *input_index,
        shape,
    });
    *input_index += volume;
    op
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
