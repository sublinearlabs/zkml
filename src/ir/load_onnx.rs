use std::fmt::Debug;
use std::ops::Deref;
use std::path::PathBuf;
use tract_core::internal::tract_itertools::Itertools;
use tract_core::ops::{einsum::EinSum, konst::Const};

use crate::ir::op::add::AddOp;
use crate::ir::op::einsum::EinsumOp;
use crate::ir::op::tensor_view::{TensorViewOp, ViewType};
use crate::ir::op::NodeOp;
use crate::ir::IR;
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
            "Source" => parse_source(node, &mut input_count),
            "Const" => parse_const(node, &mut constants),
            "EinSum" => parse_einsum(node),
            "Add" => parse_add(node),
            unknown_op => panic!("unsupported node: {}", unknown_op),
        };
        ops.push(op);
    }

    IR::new(input_count, constants, output_ids, ops)
}

fn parse_source<F: Fact, O: Debug>(node: &Node<F, O>, input_index: &mut usize) -> NodeOp {
    let shape = tract_shape_data(&node);
    let volume = shape.volume();
    let op = NodeOp::TensorView(TensorViewOp {
        id: node.id,
        tensor_type: ViewType::Input,
        start_index: *input_index,
        shape,
    });
    *input_index += volume;
    op
}

fn parse_const<F: Fact, O>(node: &Node<F, O>, constants: &mut Vec<f32>) -> NodeOp
where
    O: Debug + Clone + Deref,
    O::Target: TypedOp,
{
    let shape = tract_shape_data(&node);
    let const_tensor = &node
        .op
        .as_typed()
        .expect("op should implement typed")
        .downcast_ref::<Const>()
        .expect("failed to downcast op to const")
        .0;

    // ensure that the datum type is f32
    assert_eq!(
        const_tensor.datum_type(),
        DatumType::F32,
        "const values should only be f32"
    );
    let data = const_tensor
        .as_slice::<f32>()
        .expect("constant failed to convert to slice");

    // create op
    let op = NodeOp::TensorView(TensorViewOp {
        id: node.id,
        tensor_type: ViewType::Weights,
        start_index: constants.len(),
        shape,
    });

    constants.extend_from_slice(data);
    op
}

fn parse_einsum<F: Fact, O>(node: &Node<F, O>) -> NodeOp
where
    O: Debug + Deref,
    O::Target: TypedOp,
{
    let input_ids = input_ids(&node);
    let einsum_op = &node
        .op
        .as_typed()
        .expect("op should implement typed")
        .downcast_ref::<EinSum>()
        .expect("failed to downcast op to einsum");

    // TODO: use to_strs() to simplify the work in the einsum circuit
    NodeOp::EinSum(EinsumOp {
        id: node.id,
        input_ids,
        instruction: einsum_op.axes.to_string(),
    })
}

fn parse_add<F: Fact, O: Debug>(node: &Node<F, O>) -> NodeOp {
    let input_ids = input_ids(node);
    NodeOp::Add(AddOp {
        id: node.id,
        lhs_id: input_ids[0],
        rhs_id: input_ids[1],
    })
}

fn tract_shape_data<F: Fact, O: Debug>(node: &Node<F, O>) -> Shape {
    assert_eq!(node.outputs.len(), 1);
    let shape_data = &node.outputs[0]
        .fact
        .to_typed_fact()
        .expect("should be typed model")
        .shape;
    let shape_dims = shape_data
        .as_concrete()
        .expect("failed to get concrete shape");
    Shape::new(shape_dims.to_vec())
}

fn input_ids<F: Fact, O: Debug>(node: &Node<F, O>) -> Vec<usize> {
    node.inputs
        .iter()
        .map(|inlet| {
            // handling nodes with just one output so inlet slot should always be 0
            assert_eq!(inlet.slot, 0, "encountered inlet slot != 0");
            inlet.node
        })
        .collect_vec()
}

#[cfg(test)]
mod tests {
    use super::{load_onnx, model_graph_to_ir};
    #[test]
    fn test_load_onnx() {
        let model_graph = load_onnx("models/linear_regression.onnx".into());
        let ir = model_graph_to_ir(&model_graph);
    }
}
