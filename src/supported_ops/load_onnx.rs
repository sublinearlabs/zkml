use std::path::PathBuf;

use tract_core::ops::{binary::BinMiniOp, einsum};
use tract_onnx::prelude::*;

use crate::{supported_ops::Einsum, tensor::Shape};

use super::{parse_tract_op, Constant, Input, OpInfo, SupportedAdd, SupportedOps, TractOps};

fn load_onnx(path: PathBuf) {
    let model = tract_onnx::onnx()
        .model_for_path(path)
        .unwrap()
        .with_input_fact(0, InferenceFact::dt_shape(f32::datum_type(), tvec!(1, 1)))
        .unwrap()
        .into_typed()
        .unwrap()
        .into_decluttered()
        .unwrap();

    let mut last_input_index = 0;
    let mut ir_info = vec![];

    for node in model.nodes() {
        let op = node.op.clone().into();
        let node_info = parse_tract_op(op, node.id, &mut last_input_index);
        dbg!(&node_info);
        ir_info.push(node_info);
    }
}

#[cfg(test)]
mod test {

    use super::load_onnx;

    #[test]
    fn test_load_onnx() {
        load_onnx("models/test_onnx_model.onnx".into());
    }
}
