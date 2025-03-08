use std::path::PathBuf;

use tract_onnx::prelude::*;

use super::{parse_tract_op, SupportedOps};

pub(crate) fn load_onnx(path: PathBuf) -> Graph<TypedFact, Box<dyn TypedOp>> {
    tract_onnx::onnx()
        .model_for_path(path)
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1)))
        .unwrap()
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap()
}

pub(crate) fn model_graph_to_ir(
    model_graph: &Graph<TypedFact, Box<dyn TypedOp>>,
) -> Vec<SupportedOps> {
    let mut last_input_index = 0;
    let mut ir_info = vec![];

    for node in model_graph.nodes() {
        let op = node.op.clone().into();
        let node_info = parse_tract_op(op, node.id, &mut last_input_index);
        ir_info.push(node_info);
    }

    ir_info
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
