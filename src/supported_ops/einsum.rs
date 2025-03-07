use crate::tensor::shape::Shape;
use crate::tensor::shape_indices::ShapeIndices;
use crate::tensor::tensor::Tensor;
use std::collections::{BTreeSet, HashMap};

// TODO: add documentation
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
        let fix_insn = self.mapping.iter().map(|(source, target)| (*target, source_values[*source])).collect();
        self.shape.index_iter(Some(fix_insn))
    }
}

// TODO: add documentation
fn einsum(insn: &str, inputs: &[Tensor<usize>]) -> Tensor<usize> {
    // assumes the instruction string is valid
    // TODO: deal with unwrap
    let [input_insn, output_insn]: [&str; 2] = insn
        .split("->")
        .take(2)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    let input_insn = input_insn.split(",").collect::<Vec<_>>();

    // now that I have the input instruction and the output instruction what is next??
    // I need to do a dimension analysis
    let mut symbol_dimensions: HashMap<char, usize> = HashMap::new();
    for (inst, input_tensor) in input_insn.iter().zip(inputs.iter()) {
        for (char, dim) in inst.chars().zip(input_tensor.shape.dims.iter()) {
            symbol_dimensions.insert(char, *dim);
        }
    }

    // given the symbol dimension, we can get the shape of all inputs and all outputs
    // but what do we do with this shape??
    // the goal is to create an empty tensor with the output shape
    // then go through each index in that tensor
    // but we want to fix values in the input, based on the output shape change
    // how can we achieve this??
    // let us call free variables the variables that appear in the output
    // those variables are supposed to be fixed in the input
    // everytime we get an output index, we should link the values with the concrete index value
    // that can be done implicitly
    // next we should generate fixed iterators for the input shapes
    //

    let free_variables: BTreeSet<char> = output_insn.chars().collect();
    let free_variable_positions: HashMap<char, usize> = free_variables
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect();

    // generate output shape
    let mut output_shape = vec![0; output_insn.len()];
    for (i, char) in free_variables.iter().enumerate() {
        output_shape[i] = symbol_dimensions[&char];
    }
    let output_shape = Shape::new(output_shape);

    let mut output_tensor = Tensor::<usize>::new(None, output_shape.clone());

    // output shape is the same as free variables
    // I'd like a map from free variable to the concrete output

    // next we fill in each value for the output
    for output_index in output_shape.index_iter(None) {
        let mut fixed_input_shapes = vec![];
        for inst in input_insn.iter() {
            let mut fix_insn = vec![];
            for (i, char) in inst.chars().enumerate() {
                if let Some(pos) = free_variable_positions.get(&char) {
                    fix_insn.push((i, output_index[*pos]))
                }
            }
            fixed_input_shapes.push(fix_insn);
        }

        let mut input_iters = IndexZip::new(
            fixed_input_shapes
                .into_iter()
                .zip(inputs.iter())
                .map(|(fix_insn, tensor)| tensor.shape.index_iter(Some(fix_insn)))
                .collect::<Vec<_>>(),
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
        let shape_iter_generator = FixedShapeGenerator::from("ijk", &free_variables, Shape::new(vec![2, 5, 1]));
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
