type Shape = Vec<usize>;

// TODO: add documentation
struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: Default + Clone> Tensor<T> {
    fn new(data: Vec<T>, shape: Shape) -> Self {
        let mut t = Self::empty_from_shape(shape);
        t.data = data;
        t
    }

    fn empty_from_shape(shape: Shape) -> Self {
        Self {
            data: vec![T::default(); shape.iter().product()],
            shape,
        }
    }
}
