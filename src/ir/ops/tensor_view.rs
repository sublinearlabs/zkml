use crate::tensor::shape::Shape;

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
