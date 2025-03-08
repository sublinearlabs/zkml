use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;

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
    fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
        input: &[Variable],
        constants: &[Variable],
    ) -> Tensor<Variable> {
        let range = self.start_index..(self.start_index + self.shape.volume());
        let tensor_data = match self.tensor_type {
            ViewType::Input => input[range].to_vec(),
            ViewType::Weights => constants[range].to_vec()
        };
        Tensor::new(Some(tensor_data), self.shape.clone())
    }
}
