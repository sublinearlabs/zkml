use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI};

use crate::{quantization::quantized_float::QuantizedFloat, tensor::tensor::Tensor};

#[derive(Debug, Clone)]
pub(crate) struct MaxOp {
    pub(crate) id: usize,
}

impl MaxOp {
    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &Builder,
        history: &HashMap<usize, Tensor<QuantizedFloat>>,
    ) -> Tensor<QuantizedFloat> {
        todo!()
    }
}
