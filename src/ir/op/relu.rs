use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};

use crate::tensor::{shape::Shape, tensor::Tensor};

struct Relu {
    id: usize,
    name: String,
    input_id: usize,
}

impl Relu {
    fn new(id: usize, name: String, input_id: usize) -> Self {
        Self { id, name, input_id }
    }

    fn compute<T: Default + Clone + PartialOrd>(&self, input: Tensor<T>) -> Tensor<T> {
        let mut res = vec![];
        for val in input.data {
            if val < T::default() {
                res.push(T::default());
            } else {
                res.push(val);
            }
        }
        Tensor::new(Some(res), input.shape)
    }

    // fn create_circuit<C: Config, Builder: RootAPI<C>>(
    //     &self,
    //     api: &mut Builder,
    //     history: &HashMap<usize, Tensor<Variable>>,
    //     input_values: &Vec<Variable>,
    //     weight_value: &Vec<Variable>,
    //     lookup_table: &Vec<Variable>
    // ) -> Tensor<Variable> {
    //     let mut res_data = vec![];

    //     let lookup_table = vec![];

    //     let input_data = history.get(&self.input_id).unwrap();

    //     for input in input_data.data {
    //         let c = api.
    //     }

    //     Tensor::new(Some(res_data), input_data.shape.clone())
    // }
}

// Returns 1 if num is zero or 1, returns zero otherwise
// fn check_binary<C: Config, Builder: RootAPI<C>>(api: Builder, num: Variable) -> Variable {
//     api.add(api.sub(C::CircuitField::One, num), num)
// }



// Represent Input

// Treat (x ) as unsigned integer in finite field

// x=5

// Bit Decomposition

// Decompose (x ) into bits using gadgets
// Bits of 5: 000...0101

// Comparison
// Check if x < 2^31
// x < 2^{31}x < 2^{31}
//  using less-than gadget

// For 5, output 1 (since 5 < 2^31)

// Compute Output
// Multiply ( x )
// by indicator:
// x ⋅ [x < 231]
// x \cdot [x < 2^{31}]x \cdot [x < 2^{31}]

// Output 5 (since indicator is 1)

fn decompose_integer(base: usize, number: usize) {
    let mut res: Vec<usize> = vec![];

    let mut rem = number;

    res.push(rem % base);
    rem = rem / base;

    while rem > 0 {
        if rem < base {
            res.push(rem);
            rem = 0;
        } else {
            res.push(rem % base);
            rem = rem / base;
        }
    }
}

fn is_less_than(lhs: usize, rhs: usize) -> u8 {
    todo!()
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::{
        ir::op::relu::{decompose_integer, Relu},
        tensor::{shape::Shape, tensor::Tensor},
    };

    #[test]
    fn test_relu() {
        let mut history = HashMap::new();

        history.insert(
            0,
            Tensor::new(Some(vec![2, 0, -1, -5, 7, -2]), Shape::new(vec![1, 6])),
        );

        let relu = Relu::new(1, "relu_op".to_string(), 0);

        let input = history.get(&0).unwrap();

        let res = relu.compute(input.clone());

        dbg!(decompose_integer(2, 5));
    }
}
