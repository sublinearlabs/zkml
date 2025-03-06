use std::thread::current;

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

/// Represents the shape of a tensor
struct Shape {
    dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Shape {
    /// Instantiates a new shape (computes stride values at this point)
    fn new(dims: Vec<usize>) -> Self {
        Self {
            strides: compute_strides(&dims),
            dims,
        }
    }

    /// Compute the number of elements in the tensor
    fn volume(&self) -> usize {
        self.dims.iter().product()
    }

    /// Converts a multi-dimensional index to a flat index
    fn flat_index(&self, index: &[usize]) -> usize {
        assert!(self.dims.len() == index.len());
        // dot product
        self.strides
            .iter()
            .zip(index.iter())
            .map(|(a, b)| a * b)
            .sum()
    }
}

fn compute_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; dims.len()];
    for i in (0..dims.len() - 1).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

struct ShapeIndices {
    shape: Shape,
    current: Option<Vec<usize>>,
}

impl ShapeIndices {
    fn new(shape: Shape) -> Self {
        let current = if shape.volume() == 0 {
            None
        } else {
            Some(vec![0; shape.dims.len()])
        };
        Self { shape, current }
    }
}

impl Iterator for ShapeIndices {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        let current = self.current.as_ref()?;

        let mut next = current.clone();

        for i in (0..next.len()).rev() {
            if next[i] < self.shape.dims[i] - 1 {
                // if we can increment this position
                // we do so and return
                next[i] += 1;
                self.current = Some(next);
                return self.current.clone();
            } else {
                // reset this position and continue
                // to the next dimension
                next[i] = 0
            }
        }

        self.current = None;
        None
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::Shape;

    use super::Tensor;

    #[test]
    fn test_strides_computation() {
        assert_eq!(Shape::new(vec![2, 3, 2, 4]).strides, vec![24, 8, 4, 1]);
    }

    #[test]
    fn test_tensor_value_retrieval_and_manipulation() {
        let mut a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![3, 2]));
        assert_eq!(a.get(&[0, 0]), &1);
        assert_eq!(a.get(&[2, 0]), &5);

        // update value of a at some index
        *a.get_mut(&[2, 0]) = 20;
        assert_eq!(a.get(&[2, 0]), &20);
    }

    #[test]
    fn test_shape_index_iterator() {}
}
