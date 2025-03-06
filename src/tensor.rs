type Shape = Vec<usize>;

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
            data: vec![T::default(); shape.iter().product()],
            strides: compute_strides(&shape),
            shape,
        }
    }
}

fn compute_strides(shape: &Shape) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
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
