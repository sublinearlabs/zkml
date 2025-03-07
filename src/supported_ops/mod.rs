use tract_core::{
    internal::DimLike,
    ops::{
        binary::TypedBinOp,
        einsum::EinSum,
        konst::Const,
        math::{Add, Sub},
        source::TypedSource,
        TypedOp,
    },
    prelude::*,
};

use crate::tensor::shape::Shape;

mod einsum;
mod load_onnx;

#[derive(Debug, Clone)]
struct OpInfo {
    // Index where the Ops data starts in the input data
    start_index: usize,
    // Shape of the input
    shape: Shape,
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

#[derive(Debug, Clone)]
struct SupportedAdd {
    id: usize,
    name: String,
}

#[derive(Debug, Clone)]
struct Constant {
    id: usize,
    info: OpInfo,
    name: String,
    data: Arc<Tensor>,
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
struct Input {
    id: usize,
    info: OpInfo,
    name: String,
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

impl From<TypedBinOp> for TractBinaryOps {
    fn from(value: TypedBinOp) -> Self {
        if let Some(res) = value.0.downcast_ref::<Add>() {
            return TractBinaryOps::Add(res.clone());
        };

        if let Some(res) = value.0.downcast_ref::<Sub>() {
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
                data: constant.0,
            });
            *last_input_index += shape.volume();
            res
        }
        // TractOps::Binary(typed_bin_op) => {
        //     let supported_bin_op = typed_bin_op
        //         .0
        //         .downcast_ref::<TractBinaryOps>()
        //         .unwrap();

        //     match supported_bin_op {
        //         TractBinaryOps::Add(add) => SupportedOps::Add(SupportedAdd {
        //             id: op_id,
        //             name: add.name().to_string(),
        //         }),
        //         _ => SupportedOps::Unknown,
        //     }
        // }
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
