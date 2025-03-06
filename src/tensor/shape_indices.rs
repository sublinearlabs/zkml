use crate::tensor::shape::Shape;

pub(crate) struct ShapeIndices {
    pub(crate) shape: Shape,
    current: Option<Vec<usize>>,
}

impl ShapeIndices {
    pub(crate) fn new(shape: Shape) -> Self {
        let current = if shape.volume() == 0 {
            None
        } else {
            Some(vec![])
        };
        Self { shape, current }
    }
}

impl Iterator for ShapeIndices {
    type Item = Vec<usize>;

    fn next(&mut self) -> Option<Self::Item> {
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

#[cfg(test)]
mod tests {
    use crate::tensor::shape::Shape;

    #[test]
    fn test_shape_index_iterator() {
        let a = Shape::new(vec![5]);
        assert_eq!(
            a.index_iter().collect::<Vec<_>>(),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4]]
        );

        let a = Shape::new(vec![2, 3]);
        assert_eq!(
            a.index_iter().collect::<Vec<_>>(),
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
}
