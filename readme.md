### ONNX to Expander Circuits
This project aims to enable proving AI inference by leveraging the efficiency of the GKR protocol using the Expander Compiler Collection.

- Build your ml model with your favorite tools.
- Export as ONNX
- Get verifiable inference

### Architecture

#### Simplifying ops
- we start with the onnx model
- and pass this to tract (https://github.com/sonos/tract/tree/main)
  - they strip models on training artifacts and make them extremely optimized for inference environments. 
  - this is especially useful for us, as we just want verifiable inference.
- next we convert the tract opcodes to our IR
  - here we strip nodes of constants (e.g. weights, bias, ...) and push them into some flat array
    - useful for committing to model weights
  - we also parse any relevant instructions e.g. einsum equations

  ```rust
  pub(crate) struct AddOp {
    pub(crate) id: usize,
    pub(crate) lhs_id: usize,
    pub(crate) rhs_id: usize,
  }
  
  pub(crate) struct EinsumOp {
    pub(crate) id: usize,
    pub(crate) input_ids: Vec<usize>,
    pub(crate) instruction: String,
  }
  
  pub(crate) struct TensorViewOp {
    pub(crate) id: usize,
    pub(crate) tensor_type: ViewType,
    pub(crate) start_index: usize,
    pub(crate) shape: Shape,
  }

  #[derive(Debug, Clone)]
  pub(crate) enum ViewType {
    Input,
    Weights,
  }
  ```
- given our IR and the weights we can convert have methods on each op to generate a circuit
- we do this for each IR node generating the full model circuit

### Quantization
ML algorithms use floating point numbers for their computations. ZK circuits make use of fields. We needed a way to represent floating point numbers in the field while preserving computational structure. 

To achieve this we decided on fixed point representation. 

- first we convert f32 to i32 via fixed point encoding
- we map each i32 number unto the field 
  - positive values are represented directly, negative numbers are represented as p - a  .
  - where p = field modulus.

#### Computational Structure Preservation
- addition: 
  - a + (-b) = a + p - b = p + a - b = a - b (because p is congruent to 0 mod p)
- multiplication
  - a * (-b) = a * (p - b) = ap - ab  = 0 - ab = -ab (ap is congruent to 0 mod p)

TODO for quantization
- [ ] implement and constrain hint for integer division
  - a / b = c 
  - a = b * c + r
  - provide c and r as hint, validate constraint above + range check r 
  - 0 <= r < b

- [ ] explore accurate floating point snarks 
  - see: (https://eprint.iacr.org/2024/1842.pdf)

## Example
For sample examples, run

1. Proving a basic linear regression model that predicts an output
```rust
    cargo run --example linear_regression
```

### What is next?
- [ ] implement the rest of tract opcodes
- [ ] look into tinygrad (https://github.com/tinygrad/tinygrad) as an alternative simplification backend
