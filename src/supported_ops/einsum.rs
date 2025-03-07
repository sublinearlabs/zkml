use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::{BTreeSet, HashMap};
use expander_compiler::builder::basic::RootBuilder;
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

    // TODO: rename to compute
    fn build<T: Copy + Clone + Default, B: RootAPI<C>, C: Config, P, S>(&self, builder: &mut B, inputs: &[Tensor<T>], mut prod_fn: P, mut sum_fn: S) -> Tensor<T>
    where
        P: FnMut(&mut B, &[T]) -> T,
        S: FnMut(&mut B, &[T]) -> T,
    {
        let mut output_tensor = Tensor::new(None, self.output_shape.clone());
        for output_index in output_tensor.shape.index_iter(None) {
            // TODO: document subsection
            let base_index_map: HashMap<char, usize> = self
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

            // TODO: document this section
            let combinations = if summed_ranges.is_empty() {
                vec![vec![]]
            } else {
                summed_ranges
                    .into_iter()
                    .multi_cartesian_product()
                    .collect_vec()
            };

            // TODO: document this section
            let products: Vec<T> = combinations
                .into_iter()
                .map(|combo| {
                    let mut index_map = base_index_map.clone();
                    for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
                        index_map.insert(c, v);
                    }
                    let elements: Vec<T> = self
                        .input_str
                        .iter()
                        .zip(inputs.iter())
                        .map(|(inst, tensor)| {
                            let indices: Vec<usize> = inst.iter().map(|c| index_map[c]).collect();
                            *tensor.get(&indices)
                        })
                        .collect();
                    prod_fn(builder, &elements)
                })
                .collect();

            let value = sum_fn(builder, &products);

            *output_tensor.get_mut(&output_index) = value;
        }
        output_tensor
    }

    fn compute(&self, inputs: &[Tensor<usize>]) -> Tensor<usize> {
        let mut dummy = ();
        self.build(
            &mut dummy,
            inputs,
            |_, values| values.iter().product(),
            |_, values| values.iter().sum(),
        )
    }

    fn create_circuit<Builder: RootAPI<T>, T: Config>(&self, builder: &mut Builder, inputs: &[Tensor<Variable>]) -> Tensor<Variable> {
        self.build(
            builder,
            inputs,
            |builder, values| prod_vars(builder, values),
            |builder, values| sum_vars(builder, values)
        )
    }

    // fn compute(&self, inputs: &[Tensor<usize>]) -> Tensor<usize> {
    //     let mut output_tensor = Tensor::new(None, self.output_shape.clone());
    //
    //     for output_index in output_tensor.shape.index_iter(None) {
    //         let mut index_map: HashMap<char, usize> = self
    //             .output_str
    //             .iter()
    //             .zip(output_index.iter())
    //             .map(|(&c, &v)| (c, v))
    //             .collect();
    //
    //         let summed_keys: Vec<char> = self.summed_indices.keys().copied().collect();
    //         let summed_ranges: Vec<Vec<usize>> = summed_keys
    //             .iter()
    //             .map(|c| (0..self.summed_indices[c]).collect())
    //             .collect();
    //
    //         let value = if summed_keys.is_empty() {
    //             self.input_str
    //                 .iter()
    //                 .zip(inputs.iter())
    //                 .map(|(inst, tensor)| {
    //                     let indices: Vec<usize> = inst.iter().map(|c| index_map[&c]).collect();
    //                     tensor.get(&indices)
    //                 })
    //                 .product::<usize>()
    //         } else {
    //             let combinations = summed_ranges.into_iter().multi_cartesian_product();
    //             combinations
    //                 .map(|combo| {
    //                     for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
    //                         index_map.insert(c, v);
    //                     }
    //                     self.input_str
    //                         .iter()
    //                         .zip(inputs.iter())
    //                         .map(|(inst, tensor)| {
    //                             let indices: Vec<usize> =
    //                                 inst.iter().map(|c| index_map[&c]).collect();
    //                             tensor.get(&indices)
    //                         })
    //                         .product::<usize>()
    //                 })
    //                 .sum()
    //         };
    //
    //         *output_tensor.get_mut(&output_index) = value;
    //     }
    //
    //     output_tensor
    // }

    // TODO: clean up + remove duplicates
    // fn create_circuit<Builder: RootAPI<T>, T: Config>(
    //     &self,
    //     builder: &mut Builder,
    //     inputs: &[Tensor<Variable>],
    // ) -> Tensor<Variable> {
    //     let mut output_tensor = Tensor::new(None, self.output_shape.clone());
    //
    //     for output_index in output_tensor.shape.index_iter(None) {
    //         let mut index_map: HashMap<char, usize> = self
    //             .output_str
    //             .iter()
    //             .zip(output_index.iter())
    //             .map(|(&c, &v)| (c, v))
    //             .collect();
    //
    //         let summed_keys: Vec<char> = self.summed_indices.keys().copied().collect();
    //         let summed_ranges: Vec<Vec<usize>> = summed_keys
    //             .iter()
    //             .map(|c| (0..self.summed_indices[c]).collect())
    //             .collect();
    //
    //         let value = if summed_keys.is_empty() {
    //             let vars = self
    //                 .input_str
    //                 .iter()
    //                 .zip(inputs.iter())
    //                 .map(|(inst, tensor)| {
    //                     let indices: Vec<usize> = inst.iter().map(|c| index_map[&c]).collect();
    //                     *tensor.get(&indices)
    //                 })
    //                 .collect_vec();
    //             prod_vars(builder, vars.as_slice())
    //         } else {
    //             let combinations = summed_ranges.into_iter().multi_cartesian_product();
    //             let vars = combinations
    //                 .map(|combo| {
    //                     for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
    //                         index_map.insert(c, v);
    //                     }
    //                     let vars = self
    //                         .input_str
    //                         .iter()
    //                         .zip(inputs.iter())
    //                         .map(|(inst, tensor)| {
    //                             let indices: Vec<usize> =
    //                                 inst.iter().map(|c| index_map[&c]).collect();
    //                             *tensor.get(&indices)
    //                         })
    //                         .collect_vec();
    //                     prod_vars(builder, vars.as_slice())
    //                 })
    //                 .collect_vec();
    //             sum_vars(builder, vars.as_slice())
    //         };
    //
    //         *output_tensor.get_mut(&output_index) = value;
    //     }
    //
    //     output_tensor
    // }
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

fn sum_vars<Builder: RootAPI<T>, T: Config>(builder: &mut Builder, input: &[Variable]) -> Variable {
    // TODO: add proper error handling
    input
        .iter()
        .cloned()
        .reduce(|acc, curr| builder.add(acc, curr))
        .unwrap()
}

fn prod_vars<Builder: RootAPI<T>, T: Config>(
    builder: &mut Builder,
    input: &[Variable],
) -> Variable {
    // TODO: add proper error handling
    input
        .iter()
        .cloned()
        .reduce(|acc, curr| builder.mul(acc, curr))
        .unwrap()
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
