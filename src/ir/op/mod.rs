use crate::ir::op::add::AddOp;
use crate::ir::op::einsum::EinsumOp;
use crate::ir::op::relu::ReluOp;
use crate::ir::op::tensor_view::TensorViewOp;
use crate::quantization::quantized_float::QuantizedFloat;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;

pub(crate) mod add;
pub(crate) mod einsum;
pub(crate) mod relu;
pub(crate) mod tensor_view;

#[derive(Debug, Clone)]
pub enum NodeOp {
    Add(AddOp),
    TensorView(TensorViewOp),
    EinSum(EinsumOp),
    Relu(ReluOp),
    Unknown,
}

impl NodeOp {
    pub(crate) fn id(&self) -> usize {
        match &self {
            NodeOp::Add(op) => op.id,
            NodeOp::TensorView(op) => op.id,
            NodeOp::EinSum(op) => op.id,
            NodeOp::Relu(op) => op.id,
            _ => panic!("cannot get id for unsupported op"),
        }
    }

    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<QuantizedFloat>>,
        inputs: &[Variable],
        constants: &[Variable],
        shift: Variable,
    ) -> Tensor<QuantizedFloat> {
        match &self {
            NodeOp::Add(op) => op.create_circuit(api, history),
            NodeOp::TensorView(op) => op.create_circuit(inputs, constants),
            NodeOp::EinSum(op) => op.create_circuit(api, history, shift),
            NodeOp::Relu(op) => op.create_circuit(api, history),
            _ => panic!("cannot create circuit for unsupported op"),
        }
    }
}
