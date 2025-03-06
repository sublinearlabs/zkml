use crate::tensor::shape_indices::ShapeIndices;

#[derive(Clone)]
/// Represents the shape of a tensor
pub(crate) struct Shape {
    pub(crate) dims: Vec<usize>,
    strides: Vec<usize>,
}

impl Shape {
    /// Instantiates a new shape (computes stride values at this point)
    pub(crate) fn new(dims: Vec<usize>) -> Self {
        Self {
            strides: compute_strides(&dims),
            dims,
        }
    }

    /// Compute the number of elements in the tensor
    pub(crate) fn volume(&self) -> usize {
        self.dims.iter().product()
    }

    /// Converts a multi-dimensional index to a flat index
    pub(crate) fn flat_index(&self, index: &[usize]) -> usize {
        assert!(self.dims.len() == index.len());
        // dot product
        self.strides
            .iter()
            .zip(index.iter())
            .map(|(a, b)| a * b)
            .sum()
    }

    /// Returns an iterator that allows you to iterate over
    /// possible indexes allowed by the shape optionally constrained by
    /// some fixed_value vector
    pub(crate) fn index_iter(&self, fixed_indices: Option<Vec<(usize, usize)>>) -> ShapeIndices {
        ShapeIndices::new(self.clone(), fixed_indices)
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
    use crate::tensor::shape::Shape;

    #[test]
    fn test_strides_computation() {
        assert_eq!(Shape::new(vec![2, 3, 2, 4]).strides, vec![24, 8, 4, 1]);
    }
}
