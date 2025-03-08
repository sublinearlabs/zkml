use crate::quantization::quantized_float::QuantizedFloat;
use crate::tensor::tensor::Tensor;
use expander_compiler::frontend::{Config, RootAPI, Variable};
use std::collections::HashMap;
use tract_core::internal::tract_itertools::Itertools;

#[derive(Debug, Clone)]
pub(crate) struct AddOp {
    pub(crate) id: usize,
    pub(crate) lhs_id: usize,
    pub(crate) rhs_id: usize,
}

impl AddOp {
    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<QuantizedFloat>>,
    ) -> Tensor<QuantizedFloat> {
        let lhs = history.get(&self.lhs_id).unwrap();
        let rhs = history.get(&self.rhs_id).unwrap();

        assert_eq!(lhs.shape.volume(), rhs.shape.volume());

        let sum = lhs
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(a, b)| a.add(api, b))
            .collect_vec();

        Tensor::new(Some(sum), lhs.shape.clone())
    }
}
