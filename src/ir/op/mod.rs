use crate::ir::op::add::AddOp;
use crate::ir::op::einsum::EinsumOp;
use crate::ir::op::tensor_view::TensorViewOp;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;

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

    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
        inputs: &[Variable],
        constants: &[Variable],
    ) -> Tensor<Variable> {
        todo!()
        // match &self {
        //
        //
        // }
    }
}
