use crate::tensor::shape::Shape;

struct OpInfo {
    // Index where the Ops data starts in the input data
    start_index: usize,
    // Shape of the input
    shape: Shape
}