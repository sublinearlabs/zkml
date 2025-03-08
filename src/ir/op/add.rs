use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub(crate) struct AddOp {
    pub(crate) id: usize,
    pub(crate) lhs_id: usize,
    pub(crate) rhs_id: usize,
}

impl AddOp {
    fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
    ) -> Tensor<Variable> {
        todo!()
    }
}
