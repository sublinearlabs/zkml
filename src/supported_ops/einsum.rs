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
    // split the instruction into input and output sections
    let [input_insn, output_insn]: [&str; 2] = insn
        .split("->")
        .take(2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();
    // get the individual input instructions
    let input_insn = input_insn.split(",").collect::<Vec<_>>();

    // Map characters to dimensions
    let mut symbol_dimensions: HashMap<char, usize> = HashMap::new();
    for (inst, input_tensor) in input_insn.iter().zip(inputs.iter()) {
        for (char, dim) in inst.chars().zip(input_tensor.shape.dims.iter()) {
            symbol_dimensions.insert(char, *dim);
        }
    }

    // Output indices (free variables)
    let output_chars: Vec<char> = output_insn.chars().collect();
    let output_shape = Shape::new(output_chars.iter().map(|c| symbol_dimensions[c]).collect());

    // Find summed indices (in inputs, not in output)
    // all the output characters sorted
    let output_set: BTreeSet<char> = output_chars.clone().into_iter().collect();
    // similar to symbol dimension
    let mut summed_indices: HashMap<char, usize> = HashMap::new();
    // for each input instr
    for inst in input_insn.iter() {
        // for each symbol in any input
        for c in inst.chars() {
            // if the output does not contain it then
            if !output_set.contains(&c) {
                // push that to the summed indices with their corresponding symbols
                summed_indices.insert(c, symbol_dimensions[&c]);
            }
        }
    }

    // build the output tensor
    let mut output_tensor = Tensor::<usize>::new(None, output_shape);

    // iterate over each output index
    for output_index in output_tensor.shape.index_iter(None) {
        // if you index at a given character, you get the current concrete value for that character
        let mut index_map: HashMap<char, usize> = output_chars
            .iter()
            .zip(output_index.iter())
            .map(|(&c, &v)| (c, v))
            .collect();
        // gives you the symbol for all the indices in the input but not in the output
        let summed_keys: Vec<char> = summed_indices.keys().copied().collect();
        // for all the symbols we are supposed to sum over we get their range from (0..max_sym)
        let summed_ranges: Vec<Vec<usize>> = summed_keys
            .iter()
            .map(|c| (0..summed_indices[c]).collect())
            .collect();

        let value = if summed_keys.is_empty() {
            // No summation: compute product directly
            // in this case, there is no value in the input that is not in the output
            // what does this mean???
            // output always has values that are in the input, so this must mean input and output
            // use exactly the same symbols if you push to a set.
            // evaluate each input tensor at the location describe by the output index
            // find the product of the result
            input_insn
                .iter()
                .zip(inputs.iter())
                .map(|(inst, tensor)| {
                    let indices: Vec<usize> = inst.chars().map(|c| index_map[&c]).collect();
                    tensor.get(&indices)
                })
                .product::<usize>()
        } else {
            // Summation required: use combinations
            // for all the values we are summing over, we find the combination of the range via multi cartesian product
            let combinations = summed_ranges.into_iter().multi_cartesian_product();
            combinations
                .map(|combo| {
                    // for each combination, recall a combination represents the set of values for no in the output
                    // index map already contains the fixed values for the free variables
                    for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
                        // here we just add the current concrete values
                        index_map.insert(c, v);
                    }
                    // then we go through each input and then get out the appropriate value
                    // get the tensor value at that point and the multiply all of this
                    // finally we sum all of this to get our output
                    input_insn
                        .iter()
                        .zip(inputs.iter())
                        .map(|(inst, tensor)| {
                            let indices: Vec<usize> = inst.chars().map(|c| index_map[&c]).collect();
                            tensor.get(&indices)
                        })
                        .product::<usize>()
                })
                .sum()
        };

        // store the output value
        *output_tensor.get_mut(&output_index) = value;
    }

    output_tensor
}

#[cfg(test)]
mod tests {
    use crate::supported_ops::einsum::einsum;
    use crate::tensor::shape::Shape;
    use crate::tensor::tensor::Tensor;
    use std::collections::HashMap;

    #[test]
    fn test_einsum() {
        // Original test: Matrix multiplication
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->ik", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![36, 41, 64, 73]), Shape::new(vec![2, 2]))
        );

        // Original test: Contraction to a vector
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->k", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![100, 114]), Shape::new(vec![2]))
        );

        // New test 1: Element-wise multiplication of two vectors
        let a = Tensor::new(Some(vec![1, 2, 3]), Shape::new(vec![3]));
        let b = Tensor::new(Some(vec![4, 5, 6]), Shape::new(vec![3]));
        let result = einsum("i,i->i", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![4, 10, 18]), Shape::new(vec![3]))
        );

        // New test 2: Summing over rows and columns of a matrix
        let a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![2, 3]));
        let result = einsum("ij->i", &[a]);
        assert_eq!(result, Tensor::new(Some(vec![6, 15]), Shape::new(vec![2])));
        let a = Tensor::new(Some(vec![1, 2, 3, 4, 5, 6]), Shape::new(vec![2, 3]));
        let result = einsum("ij->j", &[a]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![5, 7, 9]), Shape::new(vec![3]))
        );

        // New test 3: Outer product of two vectors
        let a = Tensor::new(Some(vec![1, 2, 3]), Shape::new(vec![3]));
        let b = Tensor::new(Some(vec![4, 5]), Shape::new(vec![2]));
        let result = einsum("i,j->ij", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![4, 5, 8, 10, 12, 15]), Shape::new(vec![3, 2]))
        );

        // New test 4: Outer product of three vectors
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

        // New test 5: Contraction of three matrices
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
