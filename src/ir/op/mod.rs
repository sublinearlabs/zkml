use crate::ir::op::add::AddOp;
use crate::ir::op::conv::ConvOp;
use crate::ir::op::einsum::EinsumOp;
use crate::ir::op::max::MaxOp;
use crate::ir::op::maxpool::MaxPoolOp;
use crate::ir::op::relu::ReluOp;
use crate::ir::op::reshape::ReshapeOp;
use crate::ir::op::tensor_view::TensorViewOp;
use crate::quantization::quantized_float::QuantizedFloat;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;

pub(crate) mod add;
pub(crate) mod conv;
pub(crate) mod einsum;
pub(crate) mod max;
pub(crate) mod maxpool;
pub(crate) mod relu;
pub(crate) mod reshape;
pub(crate) mod tensor_view;

#[derive(Debug, Clone)]
pub enum NodeOp {
    Add(AddOp),
    TensorView(TensorViewOp),
    EinSum(EinsumOp),
    Relu(ReluOp),
    Conv(ConvOp),
    Max(MaxOp),
    MaxPool(MaxPoolOp),
    Reshape(ReshapeOp),
    Unknown,
}

impl NodeOp {
    pub(crate) fn id(&self) -> usize {
        match &self {
            NodeOp::Add(op) => op.id,
            NodeOp::TensorView(op) => op.id,
            NodeOp::EinSum(op) => op.id,
            NodeOp::Relu(op) => op.id,
            NodeOp::Conv(op) => op.id,
            NodeOp::Max(op) => op.id,
            NodeOp::MaxPool(op) => op.id,
            NodeOp::Reshape(op) => op.id,
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
            NodeOp::Conv(op) => op.create_circuit(api, history),
            NodeOp::Max(op) => op.create_circuit(api, history),
            NodeOp::MaxPool(op) => op.create_circuit(api, history),
            NodeOp::Reshape(op) => op.create_circuit(api, history),
            _ => panic!("cannot create circuit for unsupported op"),
        }
    }
}
