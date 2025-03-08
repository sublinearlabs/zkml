use crate::ir::ops::add::AddOp;
use crate::ir::ops::einsum::EinsumOp;
use crate::ir::ops::input::InputOp;
use crate::ir::ops::r#const::ConstOp;

pub(crate) mod add;
pub(crate) mod r#const;
pub(crate) mod einsum;
pub(crate) mod input;

#[derive(Debug, Clone)]
pub(crate) enum Ops {
    Add(AddOp),
    Constant(ConstOp),
    Input(InputOp),
    EinSum(EinsumOp),
    Unknown,
}
