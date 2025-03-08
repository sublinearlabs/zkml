use crate::ir::Ops;

/// Circuit friendly representation of some ML computational graph
struct IR {
    /// number of flattened inputs to some model
    input_count: usize,
    /// contains all constants in the computational graph
    /// this includes weights, bias, ...
    constants: Vec<f32>,
    /// node id's for output nodes
    output_ids: Vec<usize>,
    /// Flattened representation of the computational graph
    ops: Vec<Ops>,
}
