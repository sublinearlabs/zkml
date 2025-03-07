use crate::tensor::shape::Shape;
use crate::tensor::shape_indices::ShapeIndices;
use crate::tensor::tensor::Tensor;
use std::collections::{BTreeSet, HashMap};
use tract_core::ndarray::indices;

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

    let free_variables: BTreeSet<char> = output_insn.chars().collect();
    let free_variable_positions: HashMap<char, usize> = free_variables
        .iter()
        .enumerate()
        .map(|(i, v)| (*v, i))
        .collect();

    dbg!(&free_variable_positions);

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
    use crate::supported_ops::einsum::einsum;
    use crate::tensor::shape::Shape;
    use crate::tensor::tensor::Tensor;

    #[test]
    fn test_einsum() {
        let a = Tensor::new(Some(vec![2, 3, 4, 5]), Shape::new(vec![2, 2]));
        let b = Tensor::new(Some(vec![6, 7, 8, 9]), Shape::new(vec![2, 2]));
        let result = einsum("ij,jk->ik", &[a, b]);
        assert_eq!(
            result,
            Tensor::new(Some(vec![36, 41, 64, 73]), Shape::new(vec![2, 2]))
        );
    }
}
