## ZKML
This project aims to enable proving AI inference by leveraging the efficiency of the GKR protocol using the Expander Compiler Collection.

## Steps
1. The model is written in python and compiled to onnx.
2. The onnx file is loaded to get the computational graph
3. Every operation node in the onnx is converted to a circuit and wired together
4. The circuit is executed and proven using the Expander Compiler Collection

## Supported Operations
1. EinSum
2. Add
3. Constant
4. Input


## Example
For sample examples, run

1. Proving a basic linear regression model that predicts an output
```rust
    cargo run --package zkml --example linear_regression
```

2. Proving the prediction of house prices in California using a model trained on the California housing dataset from Scikit learn
```rust
    cargo run --package zkml --example linear_regression2
```