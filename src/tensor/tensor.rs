use crate::tensor::shape::Shape;

/// Flat representation of an n-dimensional tensor
struct Tensor<T> {
    data: Vec<T>,
    shape: Shape,
}

impl<T: Default + Clone> Tensor<T> {
    /// Instantiates a new Tensor
    /// ensures the size of data matches the shape
    /// if no data, performs default T allocation of correct size
    fn new(data: Option<Vec<T>>, shape: Shape) -> Self {
        let data = if let Some(data) = data {
            assert!(data.len() == shape.volume());
            data
        } else {
            vec![T::default(); shape.volume()]
        };

        Self { data, shape }
    }

    /// Returns the value at the given multi-dimensional tensor location
    fn get(&self, index: &[usize]) -> &T {
        &self.data[self.shape.flat_index(index)]
    }

    /// Gives you mutable access to a certain location in a tensor
    fn get_mut(&mut self, index: &[usize]) -> &mut T {
        &mut self.data[self.shape.flat_index(index)]
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::shape::Shape;
    use super::Tensor;

    #[test]
    fn test_tensor_value_retrieval_and_manipulation() {
        let mut a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![3, 2]));
        assert_eq!(a.get(&[0, 0]), &1);
        assert_eq!(a.get(&[2, 0]), &5);

        // update value of a at some index
        *a.get_mut(&[2, 0]) = 20;
        assert_eq!(a.get(&[2, 0]), &20);
    }
}
