/// Flat representation of an n-dimensional tensor
struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: Default + Clone> Tensor<T> {
    fn new(data: Option<Vec<T>>, shape: Shape) -> Self {
        let data = if let Some(data) = data {
            assert!(data.len() == shape.volume());
            data
        } else {
            vec![T::default(); shape.volume()]
        };

        Self { data, shape }
    }
}

struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Shape {
    fn new(dims: Vec<usize>) -> Self {
        Self {
            strides: compute_strides(&dims),
            dims,
        }
    }

    fn volume(&self) -> usize {
        self.dims.iter().product()
    }
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use crate::tensor::Shape;

    #[test]
    fn test_strides_computation() {
        assert_eq!(Shape::new(vec![2, 3, 2, 4]).strides, vec![24, 8, 4, 1]);
    }
}
