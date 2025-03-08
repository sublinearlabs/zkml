use crate::ir::ops::add::AddOp;
use crate::ir::ops::einsum::EinsumOp;
use crate::ir::ops::tensor_view::TensorViewOp;

pub(crate) mod add;
pub(crate) mod einsum;
pub(crate) mod tensor_view;

#[derive(Debug, Clone)]
pub(crate) enum Ops {
    Add(AddOp),
    TensorView(TensorViewOp),
    EinSum(EinsumOp),
    Unknown,
}
