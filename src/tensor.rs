struct Shape(Vec<usize>);

impl Shape {
    fn volume(&self) -> usize {
        self.0.iter().product()
    }

    fn strides(&self) -> Vec<usize> {
        let shape = &self.0;
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len() - 1).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}

// TODO: add documentation
struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
    strides: Vec<usize>,
}

impl<T: Default + Clone> Tensor<T> {
    fn new(data: Vec<T>, shape: Shape) -> Self {
        let mut t = Self::empty_from_shape(shape);
        t.data = data;
        t
    }

    fn empty_from_shape(shape: Shape) -> Self {
        Self {
            data: vec![T::default(); shape.volume()],
            strides: shape.strides(),
            shape,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Tensor;

    #[test]
    fn test_tensor_stride_construction() {
        assert_eq!(
            Tensor::<u32>::empty_from_shape(vec![2, 3, 2, 4]).strides,
            vec![24, 8, 4, 1]
        );
    }
}
