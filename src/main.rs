use inkwell::context::Context;
use anyhow::Result;

mod ast;
mod llvm_codegen;

use ast::*;
use llvm_codegen::{Codegen, CodegenContext};

fn main() -> Result<()> {
    let context = Context::create();
    let mut codegen_ctx = CodegenContext::new(&context, "test_module");

    // Create a simple test program
    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "test_function".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: AstType::Int,
                    },
                ],
                return_type: Some(AstType::Int),
                body: Block {
                    statements: vec![
                        Stmt::Let {
                            name: "y".to_string(),
                            ty: Some(AstType::Int),
                            value: Some(Expr::Binary {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::VarRef("x".to_string())),
                                rhs: Box::new(Expr::Literal(Literal::Int(42))),
                            }),
                        },
                        Stmt::Expr(Expr::VarRef("y".to_string())),
                    ],
                },
            }),
        ],
    };

    // Generate LLVM IR
    program.codegen(&mut codegen_ctx);

    // Print the generated IR
    println!("{}", codegen_ctx.module.print_to_string().to_string());

    Ok(())
}
