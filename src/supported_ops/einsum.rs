use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;
use std::collections::{BTreeSet, HashMap};
use tract_core::internal::tract_itertools::Itertools;

struct EinSum {
    input_str: Vec<Vec<char>>,
    output_str: Vec<char>,
    symbol_dimensions: HashMap<char, usize>,
    summed_indices: HashMap<char, usize>,
    output_shape: Shape,
}

impl EinSum {
    fn new(instruction: &str, input_shapes: &[Shape]) -> Self {
        let [input_insn, output_insn]: [&str; 2] = instruction
            .split("->")
            .take(2)
            .collect_vec()
            .try_into()
            .unwrap();

        let input_insn = input_insn
            .split(",")
            .map(|s| s.chars().collect_vec())
            .collect_vec();
        let output_insn = output_insn.chars().collect_vec();

        // map each character index to its dimension size
        let mut symbol_dimensions = HashMap::new();
        for (inst, shape) in input_insn.iter().zip(input_shapes.iter()) {
            for (&c, &dim) in inst.iter().zip(shape.dims.iter()) {
                symbol_dimensions.insert(c, dim);
            }
        }

        // determine the output shape
        let output_shape = Shape::new(output_insn.iter().map(|c| symbol_dimensions[c]).collect());

        // get the indices to sum over
        let mut summed_indices: HashMap<char, usize> = HashMap::new();
        for inst in input_insn.iter() {
            for c in inst {
                if !output_insn.contains(&c) {
                    summed_indices.insert(*c, symbol_dimensions[&c]);
                }
            }
        }

        Self {
            input_str: input_insn,
            output_str: output_insn,
            symbol_dimensions,
            summed_indices,
            output_shape,
        }
    }

    fn compute(&self, inputs: &[Tensor<usize>]) -> Tensor<usize> {
        let mut output_tensor = Tensor::new(None, self.output_shape.clone());

        for output_index in output_tensor.shape.index_iter(None) {
            let mut index_map: HashMap<char, usize> = self
                .output_str
                .iter()
                .zip(output_index.iter())
                .map(|(&c, &v)| (c, v))
                .collect();

            let summed_keys: Vec<char> = self.summed_indices.keys().copied().collect();
            let summed_ranges: Vec<Vec<usize>> = summed_keys
                .iter()
                .map(|c| (0..self.summed_indices[c]).collect())
                .collect();

            let value = if summed_keys.is_empty() {
                self.input_str
                    .iter()
                    .zip(inputs.iter())
                    .map(|(inst, tensor)| {
                        let indices: Vec<usize> = inst.iter().map(|c| index_map[&c]).collect();
                        tensor.get(&indices)
                    })
                    .product::<usize>()
            } else {
                let combinations = summed_ranges.into_iter().multi_cartesian_product();
                combinations
                    .map(|combo| {
                        for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
                            index_map.insert(c, v);
                        }
                        self.input_str
                            .iter()
                            .zip(inputs.iter())
                            .map(|(inst, tensor)| {
                                let indices: Vec<usize> =
                                    inst.iter().map(|c| index_map[&c]).collect();
                                tensor.get(&indices)
                            })
                            .product::<usize>()
                    })
                    .sum()
            };

            *output_tensor.get_mut(&output_index) = value;
        }

        output_tensor
    }
}

fn einsum(insn: &str, inputs: &[Tensor<usize>]) -> Tensor<usize> {
    let einsum_params = EinSum::new(
        insn,
        inputs
            .iter()
            .map(|i| i.shape.clone())
            .collect::<Vec<_>>()
            .as_slice(),
    );
    einsum_params.compute(inputs)
}

#[cfg(test)]
mod tests {
    use crate::supported_ops::einsum::einsum;
    use crate::tensor::shape::Shape;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_einsum() {
        // matmul
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->ik", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![36, 41, 64, 73]), Shape::new(vec![2, 2]))
        );

        // vector contraction
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->k", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![100, 114]), Shape::new(vec![2]))
        );

        // element wise multiplication of two vectors
        let a = Tensor::new(Some(vec![1, 2, 3]), Shape::new(vec![3]));
        let b = Tensor::new(Some(vec![4, 5, 6]), Shape::new(vec![3]));
        let result = einsum("i,i->i", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![4, 10, 18]), Shape::new(vec![3]))
        );

        // sum over rows and columns of a matrix
        let a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![2, 3]));
        let result = einsum("ij->i", &[a]);
        assert_eq!(result, Tensor::new(Some(vec![6, 15]), Shape::new(vec![2])));
        let a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![2, 3]));
        let result = einsum("ij->j", &[a]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![5, 7, 9]), Shape::new(vec![3]))
        );

        // outer product of two vectors
        let a = Tensor::new(Some(vec![1, 2, 3]), Shape::new(vec![3]));
        let b = Tensor::new(Some(vec![4, 5]), Shape::new(vec![2]));
        let result = einsum("i,j->ij", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![4, 5, 8, 10, 12, 15]), Shape::new(vec![3, 2]))
        );

        // outer product of three vectors
        let a = Tensor::new(Some(vec![1, 2]), Shape::new(vec![2]));
        let b = Tensor::new(Some(vec![3, 4]), Shape::new(vec![2]));
        let c = Tensor::new(Some(vec![5, 6]), Shape::new(vec![2]));
        let result = einsum("i,j,k->ijk", &[a, b, c]);
        assert_eq!(
            result,
            Tensor::new(
                Some(vec![15, 18, 20, 24, 30, 36, 40, 48]),
                Shape::new(vec![2, 2, 2])
            )
        );

        // contraction of three matrices
        let a = Tensor::new(Some(vec![1, 2, 3, 4]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![5, 6, 7, 8]), Shape::new(vec![2, 2]));
        let c = Tensor::new(Some(vec![9, 10, 11, 12]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk,kl->il", &[a, b, c]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![413, 454, 937, 1030]), Shape::new(vec![2, 2]))
        );
    }
}
