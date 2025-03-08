use crate::ir::OpInfo;

#[derive(Debug, Clone)]
pub(crate) struct ConstOp {
    id: usize,
    info: OpInfo,
    name: String,
}
