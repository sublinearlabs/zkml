use crate::ir::op::add::AddOp;
use crate::ir::op::einsum::EinsumOp;
use crate::ir::op::tensor_view::TensorViewOp;

pub(crate) mod add;
pub(crate) mod einsum;
pub(crate) mod tensor_view;

#[derive(Debug, Clone)]
pub(crate) enum NodeOp {
    Add(AddOp),
    TensorView(TensorViewOp),
    EinSum(EinsumOp),
    Unknown,
}

impl NodeOp {
    pub(crate) fn id(&self) -> usize {
        match &self {
            NodeOp::Add(op) => op.id,
            NodeOp::TensorView(op) => op.id,
            NodeOp::EinSum(op) => op.id,
            _ => panic!("cannot get id for unsupported op"),
        }
    }
}
