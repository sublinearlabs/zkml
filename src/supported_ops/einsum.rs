use crate::tensor::shape::Shape;
use crate::tensor::shape_indices::ShapeIndices;
use crate::tensor::tensor::Tensor;
use std::collections::{BTreeSet, HashMap};

// TODO: add documentation
fn parse_einsum_instruction(
    instruction: &str,
    inputs: &[Tensor<usize>],
) -> (Shape, Vec<FixedShapeGenerator>) {
    // partition instruction to get input instruction and output instruction
    let [input_insn, output_insn]: [&str; 2] = instruction
        .split("->")
        .take(2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    // get the dimension variable string for each input
    let input_insn = input_insn.split(",").collect::<Vec<_>>();

    // for each dimension variable figure out the concrete shape
    let mut symbol_dimensions: HashMap<char, usize> = HashMap::new();
    for (inst, input_tensor) in input_insn.iter().zip(inputs.iter()) {
        for (char, dim) in inst.chars().zip(input_tensor.shape.dims.iter()) {
            symbol_dimensions.insert(char, *dim);
        }
    }

    // free variables represent variables in the output
    let free_variables: BTreeSet<char> = output_insn.chars().collect();
    let free_variable_positions: HashMap<char, usize> = free_variables
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect();

    // TODO: handle empty output section here
    // generate output shape and empty output tensor
    let mut output_shape = vec![0; output_insn.len()];
    for (i, char) in free_variables.iter().enumerate() {
        output_shape[i] = symbol_dimensions[&char];
    }
    let output_shape = Shape::new(output_shape);

    // create input to free variable mappings for all inputs
    let input_iter_generators = input_insn
        .iter()
        .zip(inputs.iter())
        .map(|(input_str, input_tensor)| {
            FixedShapeGenerator::from(
                input_str,
                &free_variable_positions,
                input_tensor.shape.clone(),
            )
        })
        .collect::<Vec<_>>();

    (output_shape, input_iter_generators)
}

// TODO: add documentation
fn einsum(insn: &str, inputs: &[Tensor<usize>]) -> Tensor<usize> {
    let (output_shape, input_iter_generators) = parse_einsum_instruction(insn, inputs);
    let mut output_tensor = Tensor::<usize>::new(None, output_shape);

    // next we fill in each value for the output
    for output_index in output_tensor.shape.index_iter(None) {
        let input_iters = IndexZip::new(
            input_iter_generators
                .iter()
                .map(|gen| gen.get_iter(&output_index))
                .collect(),
        );

        *output_tensor.get_mut(&output_index) = input_iters
            .map(|indices| {
                indices
                    .iter()
                    .zip(inputs.iter())
                    .map(|(index, tensor)| tensor.get(index))
                    .product::<usize>()
            })
            .sum::<usize>()
    }

    // next we need to evaluate each input tensor at the production of their input indices
    // once we do this we mul the result
    output_tensor
}

/// Used to hold free variables to input variable mapping
/// Can produce a shape indices iterator with fixed values based on mapping
struct FixedShapeGenerator {
    // Vec<(source, target)>
    mapping: Vec<(usize, usize)>,
    shape: Shape,
}

impl FixedShapeGenerator {
    fn from(input_str: &str, free_variable_positions: &HashMap<char, usize>, shape: Shape) -> Self {
        let mut mapping = vec![];
        for (target, var) in input_str.chars().enumerate() {
            if let Some(source) = free_variable_positions.get(&var) {
                mapping.push((*source, target));
            }
        }
        Self { mapping, shape }
    }

    fn get_iter(&self, source_values: &[usize]) -> ShapeIndices {
        let fix_insn = self
            .mapping
            .iter()
            .map(|(source, target)| (*target, source_values[*source]))
            .collect();
        self.shape.index_iter(Some(fix_insn))
    }
}

struct IndexZip {
    iterators: Vec<ShapeIndices>,
}

impl IndexZip {
    fn new(iterators: Vec<ShapeIndices>) -> Self {
        Self { iterators }
    }
}

impl Iterator for IndexZip {
    type Item = Vec<Vec<usize>>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iterators
            .iter_mut()
            .map(|i| i.next())
            .collect::<Option<Vec<_>>>()
    }
}

#[cfg(test)]
mod tests {
    use crate::supported_ops::einsum::{einsum, FixedShapeGenerator};
    use crate::tensor::shape::Shape;
    use crate::tensor::tensor::Tensor;
    use std::collections::HashMap;

    #[test]
    fn test_fixed_shape_generator() {
        // assume einsum instruction ij,jk->ik
        // free variables should be [i->0, k->1]
        // for the first input ij we expect the mapping [(0,0)]
        // for the second input jk we expect the mapping [(1,1)]

        let free_variables: HashMap<char, usize> = vec![('i', 0), ('k', 1)].into_iter().collect();
        let mapping_1 =
            FixedShapeGenerator::from("ij", &free_variables, Shape::new(vec![1])).mapping;
        assert_eq!(mapping_1, vec![(0, 0)]);
        let mapping_2 =
            FixedShapeGenerator::from("jk", &free_variables, Shape::new(vec![1])).mapping;
        assert_eq!(mapping_2, vec![(1, 1)]);
    }

    #[test]
    fn test_fixed_shape_iteration() {
        // assume we have some input string ijk
        // and we have some free variable j that comes from index 0 of source
        let free_variables: HashMap<char, usize> = vec![('j', 0)].into_iter().collect();
        let shape_iter_generator =
            FixedShapeGenerator::from("ijk", &free_variables, Shape::new(vec![2, 5, 1]));
        assert_eq!(shape_iter_generator.mapping, &[(0, 1)]);
        // this should create a generator that fixes j to 3
        let iter = shape_iter_generator.get_iter(&[3]);
        let indices = iter.collect::<Vec<_>>();
        assert_eq!(indices, vec![vec![0, 3, 0], vec![1, 3, 0]]);
    }

    #[test]
    fn test_einsum() {
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->ik", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![36, 41, 64, 73]), Shape::new(vec![2, 2]))
        );

        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->k", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![36, 41, 64, 73]), Shape::new(vec![2, 2]))
        );
    }
}
