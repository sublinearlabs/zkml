use crate::ir::OpInfo;

#[derive(Debug, Clone)]
pub(crate) struct InputOp {
    pub(crate) id: usize,
    pub(crate) info: OpInfo,
    pub(crate) name: String,
}
