use inkwell::{context::Context, module::Module, builder::Builder, values::{BasicValueEnum, PointerValue, FunctionValue}};
use std::collections::HashMap;

use crate::ast::{Type, Program, Item, ItemFunction, ItemStruct, Block, Stmt, Expr, Literal, BinOp};

pub mod context;
pub mod gen_type;
pub mod gen_item;
pub mod gen_expr;
pub mod gen_stmt;

pub use context::CodegenContext;

pub trait Codegen<'ctx> {
    type Output;
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) -> Self::Output;
}

impl<'ctx> Codegen<'ctx> for Program {
    type Output = ();
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) {
        for item in &self.items {
            item.codegen(ctx);
        }
    }
} 