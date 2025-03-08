use expander_compiler::field::BN254;
use tract_core::internal::tract_itertools::Itertools;
use crate::ir::op::NodeOp;
use crate::quantization::quantizer::Quantizer;

pub(crate) mod load_onnx;
pub(crate) mod op;

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

    /// Returns the IR constants in quantized form (so we can perform operations in the field)
    pub(crate) fn quantize_constants<const N: u8>(&self) -> Vec<BN254> {
        let quantizer = Quantizer::<N>{};
        self.constants.iter().map(|c| quantizer.quantize(*c)).collect_vec()
    }
}
