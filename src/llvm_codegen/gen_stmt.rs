use super::{Codegen, CodegenContext};
use crate::ast::{Stmt, Type};
use inkwell::{
    types::BasicType,
    values::BasicValue,
};

impl<'ctx> Codegen<'ctx> for Stmt {
    type Output = ();
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) {
        match self {
            Stmt::Let { name, ty, value } => {
                let var_type = ty.clone().unwrap_or(Type::Int);  // Default to Int if type not specified
                let llvm_ty = ctx.llvm_type(&var_type);
                let alloc = ctx.builder.build_alloca(llvm_ty, name).unwrap();
                if let Some(v) = value {
                    let val = v.codegen(ctx);
                    ctx.builder.build_store(alloc, val).unwrap();
                }
                ctx.insert_variable(name, alloc, var_type);
            }
            Stmt::Expr(e) => {
                let _val = e.codegen(ctx);
            }
            Stmt::Semi(e) => {
                let _val = e.codegen(ctx);
            }
            Stmt::Return(value) => {
                if let Some(expr) = value {
                    let val = expr.codegen(ctx);
                    ctx.builder.build_return(Some(&val)).unwrap();
                } else {
                    ctx.builder.build_return(None).unwrap();
                }
            }
            Stmt::While { condition, body } => {
                let parent = ctx.builder.get_insert_block().unwrap().get_parent().unwrap();
                let cond_block = ctx.context.append_basic_block(parent, "while.cond");
                let body_block = ctx.context.append_basic_block(parent, "while.body");
                let end_block = ctx.context.append_basic_block(parent, "while.end");

                // Branch to condition block
                ctx.builder.build_unconditional_branch(cond_block).unwrap();
                ctx.builder.position_at_end(cond_block);

                // Generate condition
                let cond_val = condition.codegen(ctx).into_int_value();
                ctx.builder.build_conditional_branch(cond_val, body_block, end_block).unwrap();

                // Generate body
                ctx.builder.position_at_end(body_block);
                let _body_val = body.codegen(ctx);
                ctx.builder.build_unconditional_branch(cond_block).unwrap();

                // Continue from end block
                ctx.builder.position_at_end(end_block);
            }
            Stmt::For { var, iterator, body } => {
                // For now, assume iterator is a range expression
                // TODO: Implement proper iterator protocol
                let _val = iterator.codegen(ctx);
                let _body_val = body.codegen(ctx);
            }
        }
    }
} 