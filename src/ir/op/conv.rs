use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI};

use crate::{quantization::quantized_float::QuantizedFloat, tensor::tensor::Tensor};

#[derive(Debug, Clone)]
pub(crate) struct ConvOp {
    pub(crate) id: usize,
}

impl ConvOp {
    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &Builder,
        history: &HashMap<usize, Tensor<QuantizedFloat>>,
    ) -> Tensor<QuantizedFloat> {
        todo!()
    }
}
