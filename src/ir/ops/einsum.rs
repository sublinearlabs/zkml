use crate::tensor::shape::Shape;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, Define, RootAPI, Variable};
use std::collections::HashMap;
use tract_core::internal::tract_itertools::Itertools;

#[derive(Debug, Default, Clone)]
pub(crate) struct EinsumOp {
    pub(crate) id: usize,
    pub(crate) input_ids: Vec<usize>,
    pub(crate) instruction: String,
}

#[derive(Debug, Default, Clone)]
pub(crate) struct EinsumParams {
    input_str: Vec<Vec<char>>,
    output_str: Vec<char>,
    symbol_dimensions: HashMap<char, usize>,
    summed_indices: HashMap<char, usize>,
    output_shape: Shape,
}

impl EinsumOp {
    fn new(instruction: &str, input_shapes: &[Shape]) -> EinsumParams {
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

        EinsumParams {
            input_str: input_insn,
            output_str: output_insn,
            symbol_dimensions,
            summed_indices,
            output_shape,
        }
    }
}

impl EinsumParams {
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

    // TODO: clean up + remove duplicates
    fn create_circuit<Builder: RootAPI<T>, T: Config>(
        &self,
        builder: &mut Builder,
        inputs: &[Tensor<Variable>],
    ) -> Tensor<Variable> {
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
                let vars = self
                    .input_str
                    .iter()
                    .zip(inputs.iter())
                    .map(|(inst, tensor)| {
                        let indices: Vec<usize> = inst.iter().map(|c| index_map[&c]).collect();
                        *tensor.get(&indices)
                    })
                    .collect_vec();
                prod_vars(builder, vars.as_slice())
            } else {
                let combinations = summed_ranges.into_iter().multi_cartesian_product();
                let vars = combinations
                    .map(|combo| {
                        for (&c, &v) in summed_keys.iter().zip(combo.iter()) {
                            index_map.insert(c, v);
                        }
                        let vars = self
                            .input_str
                            .iter()
                            .zip(inputs.iter())
                            .map(|(inst, tensor)| {
                                let indices: Vec<usize> =
                                    inst.iter().map(|c| index_map[&c]).collect();
                                *tensor.get(&indices)
                            })
                            .collect_vec();
                        prod_vars(builder, vars.as_slice())
                    })
                    .collect_vec();
                sum_vars(builder, vars.as_slice())
            };

            *output_tensor.get_mut(&output_index) = value;
        }

        output_tensor
    }
}

fn einsum(insn: &str, inputs: &[Tensor<usize>]) -> Tensor<usize> {
    let einsum_params = EinsumOp::new(
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
    input
        .iter()
        .cloned()
        .reduce(|acc, curr| builder.mul(acc, curr))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use crate::ir::ops::einsum::{einsum, EinsumOp, EinsumParams};
    use crate::tensor::shape::Shape;
    use crate::tensor::tensor::Tensor;
    use expander_compiler::declare_circuit;
    use expander_compiler::field::M31;
    use expander_compiler::frontend::{
        compile, CompileOptions, Define, M31Config, RootAPI, Variable,
    };

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

    #[test]
    fn test_einsum_circuit() {
        // matmul circuit
        declare_circuit!(EinsumCircuit {
            a: [Variable; 4],
            b: [Variable; 4],
            target: [Variable; 4],
            einsum_params: EinsumParams
        });

        impl Define<M31Config> for EinsumCircuit<Variable> {
            fn define<Builder: RootAPI<M31Config>>(&self, api: &mut Builder) {
                let a_tensor = Tensor::new(Some(self.a.to_vec()), Shape::new(vec![2, 2]));
                let b_tensor = Tensor::new(Some(self.b.to_vec()), Shape::new(vec![2, 2]));
                let result = self
                    .einsum_params
                    .create_circuit(api, &[a_tensor, b_tensor]);
                for (i, v) in self.target.iter().enumerate() {
                    api.assert_is_equal(result.data[i], v);
                }
            }
        }

        impl EinsumCircuit<Variable> {
            fn new(params: EinsumParams) -> Self {
                let mut circuit = Self::default();
                circuit.einsum_params = params;
                circuit
            }
        }

        let params = EinsumOp::new(
            "ij,jk->ik",
            &[Shape::new(vec![2, 2]), Shape::new(vec![2, 2])],
        );

        let compile_result =
            compile(&EinsumCircuit::new(params), CompileOptions::default()).unwrap();
        let assignment = EinsumCircuit::<M31> {
            a: [M31::from(2), M31::from(3), M31::from(4), M31::from(5)],
            b: [M31::from(6), M31::from(7), M31::from(8), M31::from(9)],
            target: [M31::from(36), M31::from(41), M31::from(64), M31::from(73)],
            einsum_params: EinsumParams::default(),
        };
        let witness = compile_result
            .witness_solver
            .solve_witness(&assignment)
            .unwrap();
        let run_result = compile_result.layered_circuit.run(&witness);
        assert!(run_result.iter().all(|v| *v));
    }
}
