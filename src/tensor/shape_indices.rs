use crate::tensor::shape::Shape;

#[derive(Clone)]
/// Holds state for shape index iteration
pub(crate) struct ShapeIndices {
    pub(crate) shape: Shape,
    current: Option<Vec<usize>>,
    /// Fixes the value of a particular dimension
    /// Vec<(index, fixed_value)>
    fixed_indices: Vec<(usize, usize)>,
}

impl ShapeIndices {
    pub(crate) fn new(shape: Shape, fixed_indices: Option<Vec<(usize, usize)>>) -> Self {
        let current = if shape.volume() == 0 {
            None
        } else {
            Some(vec![])
        };

        Self {
            shape,
            current,
            fixed_indices: fixed_indices.unwrap_or(vec![]),
        }
    }

    /// Given the current index state, compute the next one
    /// Returns None if last state has been reached
    fn next_inner(&mut self) -> Option<Vec<usize>> {
        let current = self.current.as_ref()?;

        if current.is_empty() {
            // first iteration
            self.current = Some(vec![0; self.shape.dims.len()]);
            return self.current.clone();
        }

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

impl Iterator for ShapeIndices {
    type Item = Vec<usize>;

    // NOTE: naive implementation of index iteration with fixed values
    // can be more exact rather than guessing
    fn next(&mut self) -> Option<Self::Item> {
        let mut maybe_next = self.next_inner();

        while let (to_check) = maybe_next.as_ref()? {
            let is_match = self
                .fixed_indices
                .iter()
                .map(|(index, expected_val)| to_check[*index] == *expected_val)
                .all(|v| v);

            if is_match {
                return maybe_next;
            }

            maybe_next = self.next_inner();
        }

        maybe_next
    }
}

#[cfg(test)]
mod tests {
    use crate::tensor::shape::Shape;

    #[test]
    fn test_shape_index_iterator() {
        let a = Shape::new(vec![5]);
        assert_eq!(
            a.index_iter(None).collect::<Vec<_>>(),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );

        let a = Shape::new(vec![2, 3]);
        assert_eq!(
            a.index_iter(None).collect::<Vec<_>>(),
            vec![
                vec![0, 0],
                vec![0, 1],
                vec![0, 2],
                vec![1, 0],
                vec![1, 1],
                vec![1, 2]
            ]
        );
    }

    #[test]
    fn test_shape_index_iterator_with_fixed_values() {
        let a = Shape::new(vec![2, 3, 3]);
        assert_eq!(
            // for any result [a, b, c]
            // fix a = 1 and c = 2
            // to get [1, b, 2] for every b
            a.index_iter(Some(vec![(0, 1), (2, 2)])).collect::<Vec<_>>(),
            vec![vec![1, 0, 2], vec![1, 1, 2], vec![1, 2, 2],]
        );

        let a = Shape::new(vec![2, 3, 3]);
        assert_eq!(
            a.index_iter(Some(vec![(0, 0)])).collect::<Vec<_>>(),
            vec![
                vec![0, 0, 0],
                vec![0, 0, 1],
                vec![0, 0, 2],
                vec![0, 1, 0],
                vec![0, 1, 1],
                vec![0, 1, 2],
                vec![0, 2, 0],
                vec![0, 2, 1],
                vec![0, 2, 2],
            ]
        );
    }
}
