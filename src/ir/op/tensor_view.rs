use crate::quantization::quantized_float::QuantizedFloat;
use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;
use tract_core::internal::tract_itertools::Itertools;

#[derive(Debug, Clone)]
pub(crate) struct TensorViewOp {
    pub(crate) id: usize,
    pub(crate) tensor_type: ViewType,
    pub(crate) start_index: usize,
    pub(crate) shape: Shape,
}

#[derive(Debug, Clone)]
pub(crate) enum ViewType {
    Input,
    Weights,
}

impl TensorViewOp {
    pub(crate) fn create_circuit(
        &self,
        input: &[Variable],
        constants: &[Variable],
    ) -> Tensor<QuantizedFloat> {
        let range = self.start_index..(self.start_index + self.shape.volume());
        let tensor_data = match self.tensor_type {
            ViewType::Input => input[range].to_vec(),
            ViewType::Weights => constants[range].to_vec(),
        }
        .into_iter()
        .map(QuantizedFloat::new)
        .collect_vec();
        Tensor::new(Some(tensor_data), self.shape.clone())
    }
}
