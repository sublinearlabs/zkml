/// Flat representation of an n-dimensional tensor
struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
    strides: Vec<usize>,
}

impl<T: Default + Clone> Tensor<T> {
    fn new(data: Option<Vec<T>>, shape: Shape) -> Self {
        let data = if let Some(data) = data {
            assert!(data.len() == shape.volume());
            data
        } else {
            vec![T::default(); shape.volume()]
        };

        Self {
            data,
            strides: shape.strides(),
            shape,
        }
    }
}

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

#[cfg(test)]
mod tests {
    use crate::tensor::{Shape, Tensor};

    #[test]
    fn test_tensor_stride_construction() {
        assert_eq!(
            Tensor::<u32>::new(None, Shape(vec![2, 3, 2, 4])).strides,
            vec![24, 8, 4, 1]
        );
    }
}
