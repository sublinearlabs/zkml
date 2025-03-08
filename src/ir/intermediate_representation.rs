use crate::ir::NodeOp;

#[derive(Debug)]
/// Circuit friendly representation of some ML computational graph
pub(crate) struct IR {
    /// number of flattened inputs to some model
    input_count: usize,
    /// contains all constants in the computational graph
    /// this includes weights, bias, ...
    constants: Vec<f32>,
    /// node id's for output nodes
    output_ids: Vec<usize>,
    /// Flattened representation of the computational graph
    ops: Vec<NodeOp>,
}

impl IR {
    /// Instantiate IR
    pub(crate) fn new(
        input_count: usize,
        constants: Vec<f32>,
        output_ids: Vec<usize>,
        ops: Vec<NodeOp>,
    ) -> Self {
        IR {
            input_count,
            constants,
            output_ids,
            ops,
        }
    }
}
