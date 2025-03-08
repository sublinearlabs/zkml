use std::collections::HashMap;

use crate::ir::ops::Ops;
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

use crate::tensor::{shape::Shape, tensor::Tensor};

mod intermediate_representation;
pub(crate) mod load_onnx;
pub(crate) mod ops;

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

impl Ops {
    pub(crate) fn create_circuit<C: Config, Builder: RootAPI<C>>(
        &self,
        api: &mut Builder,
        history: &HashMap<usize, Tensor<Variable>>,
        input_value: &Vec<Variable>,
    ) -> Tensor<Variable> {
        todo!()
        // match self {
        //     Ops::Add(supported_add) => {
        //         let lhs = history.get(&supported_add.lhs_id).unwrap();
        //         let rhs = history.get(&supported_add.rhs_id).unwrap();
        //
        //         assert_eq!(lhs.shape.volume(), rhs.shape.volume());
        //
        //         let mut res_data = vec![];
        //
        //         for i in 0..lhs.shape.volume() {
        //             res_data.push(api.add(lhs.data[i], rhs.data[i]))
        //         }
        //
        //         Tensor::new(Some(res_data), lhs.shape.clone())
        //     }
        //     Ops::Input(input) => {
        //         let data = input_value[input.info.start_index..input.info.shape.volume()].to_vec();
        //         Tensor::new(Some(data), input.info.shape.clone())
        //     }
        //     Ops::Constant(constant) => todo!(),
        //     Ops::EinSum(einsum) => todo!(),
        //     Ops::Unknown => todo!(),
        // }
    }

    pub(crate) fn id(&self) -> usize {
        todo!()
        // match self {
        //     Ops::Add(supported_add) => supported_add.id,
        //     Ops::Constant(constant) => constant.id,
        //     Ops::Input(input) => input.id,
        //     Ops::EinSum(einsum) => einsum.id,
        //     Ops::Unknown => panic!("Failed trying to get Id for Unknown Op"),
        // }
    }
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

pub(crate) fn parse_tract_op(op: TractOps, op_id: usize, last_input_index: &mut usize) -> Ops {
    todo!()
    // let res = match op {
    //     TractOps::EinSum(ein_sum) => Ops::EinSum(EinsumOp {
    //         name: ein_sum.name().into_owned(),
    //         id: op_id,
    //         instruction: ein_sum.axes.to_string(),
    //         input_count: ein_sum.axes.input_count(),
    //         output_count: ein_sum.axes.output_count(),
    //     }),
    //     TractOps::Input(typed_source) => {
    //         let shape = Shape::new(get_shape_array_from_shapefact(&typed_source.fact.shape));
    //         let res = Ops::Input(InputOp {
    //             id: op_id,
    //             info: OpInfo::new(*last_input_index, shape.clone()),
    //             name: typed_source.name().to_string(),
    //         });
    //         *last_input_index += shape.volume();
    //         res
    //     }
    //     TractOps::Constant(constant) => {
    //         let shape = Shape::new(constant.0.shape().to_vec());
    //         let res = Ops::Constant(ConstOp {
    //             id: op_id,
    //             info: OpInfo::new(*last_input_index, shape.clone()),
    //             name: constant.name().to_string(),
    //             // data: constant.0,
    //         });
    //         *last_input_index += shape.volume();
    //         res
    //     }
    //     TractOps::Binary(typed_bin_op) => {
    //         let supported_bin_op: TractBinaryOps = typed_bin_op.0.into();
    //
    //         match supported_bin_op {
    //             TractBinaryOps::Add(add) => Ops::Add(AddOp {
    //                 id: op_id,
    //                 name: add.name().to_string(),
    //                 // todo!(): FIX
    //                 // Fetch info from tract
    //                 lhs_id: todo!(),
    //                 rhs_id: todo!(),
    //             }),
    //             _ => Ops::Unknown,
    //         }
    //     }
    //     _ => Ops::Unknown,
    // };
    // res
}

fn get_shape_array_from_shapefact(shape_fact: &ShapeFact) -> Vec<usize> {
    shape_fact
        .dims()
        .to_vec()
        .iter()
        .map(|val| val.to_usize().unwrap())
        .collect()
}
