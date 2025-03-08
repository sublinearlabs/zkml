use std::collections::HashMap;

use expander_compiler::frontend::{Config, RootAPI, Variable};
use tract_core::{
    internal::DimLike,
    model::ShapeFact,
    ops::{
        binary::{BinMiniOp, TypedBinOp},
        einsum::EinSum,
        konst::Const,
        math::{Add, Sub},
        source::TypedSource,
        Op, TypedOp,
    },
};

use crate::tensor::{
    shape::{self, Shape},
    tensor::Tensor,
};

mod einsum;
mod intermediate_representation;
pub(crate) mod load_onnx;

#[derive(Debug, Clone)]
pub(crate) struct OpInfo {
    // Index where the Ops data starts in the input data
    pub(crate) start_index: usize,
    // Shape of the input
    pub(crate) shape: Shape,
}

impl OpInfo {
    fn new(start_index: usize, shape: Shape) -> Self {
        Self { start_index, shape }
    }
}

#[derive(Debug, Clone)]
pub(crate) enum SupportedOps {
    Add(SupportedAdd),
    Constant(Constant),
    Input(Input),
    EinSum(Einsum),
    Unknown,
}

impl SupportedOps {
    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
        input_value: &Vec<Variable>,
    ) -> Tensor<Variable> {
        match self {
            SupportedOps::Add(supported_add) => {
                let lhs = history.get(&supported_add.lhs_id).unwrap();
                let rhs = history.get(&supported_add.rhs_id).unwrap();

                assert_eq!(lhs.shape.volume(), rhs.shape.volume());

                let mut res_data = vec![];

                for i in 0..lhs.shape.volume() {
                    res_data.push(api.add(lhs.data[i], rhs.data[i]))
                }

                Tensor::new(Some(res_data), lhs.shape.clone())
            }
            SupportedOps::Input(input) => {
                let data = input_value[input.info.start_index..input.info.shape.volume()].to_vec();
                Tensor::new(Some(data), input.info.shape.clone())
            }
            SupportedOps::Constant(constant) => todo!(),
            SupportedOps::EinSum(einsum) => todo!(),
            SupportedOps::Unknown => todo!(),
        }
    }

    pub(crate) fn get_op_id(&self) -> usize {
        match self {
            SupportedOps::Add(supported_add) => supported_add.id,
            SupportedOps::Constant(constant) => constant.id,
            SupportedOps::Input(input) => input.id,
            SupportedOps::EinSum(einsum) => einsum.id,
            SupportedOps::Unknown => panic!("Failed trying to get Id for Unknown Op"),
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct SupportedAdd {
    pub(crate) id: usize,
    pub(crate) name: String,
    pub(crate) lhs_id: usize,
    pub(crate) rhs_id: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct Constant {
    id: usize,
    info: OpInfo,
    name: String,
    // data: Tensor<>,
}

#[derive(Debug, Clone)]
pub(crate) struct Input {
    pub(crate) id: usize,
    pub(crate) info: OpInfo,
    pub(crate) name: String,
}

#[derive(Debug, Clone)]
pub(crate) struct Einsum {
    name: String,
    id: usize,
    instruction: String,
    input_count: usize,
    output_count: usize,
}

#[derive(Debug, Clone)]
pub enum TractOps {
    Input(TypedSource),
    Constant(Const),
    Binary(TypedBinOp),
    EinSum(EinSum),
    Unknown,
}

impl From<Box<dyn TypedOp>> for TractOps {
    fn from(value: Box<dyn TypedOp>) -> Self {
        if let Some(res) = value.downcast_ref::<TypedBinOp>() {
            return TractOps::Binary(res.clone());
        };

        if let Some(res) = value.downcast_ref::<Const>() {
            return TractOps::Constant(res.clone());
        };

        if let Some(res) = value.downcast_ref::<TypedSource>() {
            return TractOps::Input(res.clone());
        };

        if let Some(res) = value.downcast_ref::<EinSum>() {
            return TractOps::EinSum(res.clone());
        };

        println!("Unknown op: {:#?}", value);
        TractOps::Unknown
    }
}

#[derive(Debug, Clone)]
pub enum TractBinaryOps {
    Add(Add),
    Sub(Sub),
    Unknown,
}

impl From<Box<dyn BinMiniOp>> for TractBinaryOps {
    fn from(value: Box<dyn BinMiniOp>) -> Self {
        if let Some(res) = value.downcast_ref::<Add>() {
            return TractBinaryOps::Add(res.clone());
        };

        if let Some(res) = value.downcast_ref::<Sub>() {
            return TractBinaryOps::Sub(res.clone());
        };

        println!("Unknown binary op: {:#?}", value);
        TractBinaryOps::Unknown
    }
}

pub(crate) fn parse_tract_op(
    op: TractOps,
    op_id: usize,
    last_input_index: &mut usize,
) -> SupportedOps {
    let res = match op {
        TractOps::EinSum(ein_sum) => SupportedOps::EinSum(Einsum {
            name: ein_sum.name().into_owned(),
            id: op_id,
            instruction: ein_sum.axes.to_string(),
            input_count: ein_sum.axes.input_count(),
            output_count: ein_sum.axes.output_count(),
        }),
        TractOps::Input(typed_source) => {
            let shape = Shape::new(get_shape_array_from_shapefact(&typed_source.fact.shape));
            let res = SupportedOps::Input(Input {
                id: op_id,
                info: OpInfo::new(*last_input_index, shape.clone()),
                name: typed_source.name().to_string(),
            });
            *last_input_index += shape.volume();
            res
        }
        TractOps::Constant(constant) => {
            let shape = Shape::new(constant.0.shape().to_vec());
            let res = SupportedOps::Constant(Constant {
                id: op_id,
                info: OpInfo::new(*last_input_index, shape.clone()),
                name: constant.name().to_string(),
                // data: constant.0,
            });
            *last_input_index += shape.volume();
            res
        }
        TractOps::Binary(typed_bin_op) => {
            let supported_bin_op: TractBinaryOps = typed_bin_op.0.into();

            match supported_bin_op {
                TractBinaryOps::Add(add) => SupportedOps::Add(SupportedAdd {
                    id: op_id,
                    name: add.name().to_string(),
                    // todo!(): FIX
                    // Fetch info from tract
                    lhs_id: todo!(),
                    rhs_id: todo!(),
                }),
                _ => SupportedOps::Unknown,
            }
        }
        _ => SupportedOps::Unknown,
    };
    res
}

fn get_shape_array_from_shapefact(shape_fact: &ShapeFact) -> Vec<usize> {
    shape_fact
        .dims()
        .to_vec()
        .iter()
        .map(|val| val.to_usize().unwrap())
        .collect()
}
