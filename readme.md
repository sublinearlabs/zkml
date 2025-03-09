### ONNX to Expander Circuits

// what do I want to write here?
// we need to talk about the motivation for doing this
// we need to talk about what success looks like
// we need to talk about technical details
// overall architecture
// quantization
// what is left


The goal is

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
