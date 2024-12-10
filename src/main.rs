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
                        ty: Type::Int,
                    },
                ],
                return_type: Some(Type::Int),
                body: Block {
                    statements: vec![
                        Stmt::Let {
                            name: "y".to_string(),
                            ty: Some(Type::Int),
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variable_load() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Function(ItemFunction {
                    name: "test_var".to_string(),
                    params: vec![
                        FunctionParam {
                            name: "x".to_string(),
                            ty: Type::Int,
                        },
                    ],
                    return_type: Some(Type::Int),
                    body: Block {
                        statements: vec![
                            // Just load and return x
                            Stmt::Expr(Expr::VarRef("x".to_string())),
                        ],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        println!("Generated IR:\n{}", ir);
        
        // Basic verification
        assert!(ir.contains("define i64 @test_var(i64"));
        assert!(ir.contains("load i64, ptr %x"));
    }

    #[test]
    fn test_simple_arithmetic() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Function(ItemFunction {
                    name: "add_const".to_string(),
                    params: vec![
                        FunctionParam {
                            name: "x".to_string(),
                            ty: Type::Int,
                        },
                    ],
                    return_type: Some(Type::Int),
                    body: Block {
                        statements: vec![
                            // Just add x + 1
                            Stmt::Expr(Expr::Binary {
                                op: BinOp::Add,
                                lhs: Box::new(Expr::VarRef("x".to_string())),
                                rhs: Box::new(Expr::Literal(Literal::Int(1))),
                            }),
                        ],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        println!("Generated IR:\n{}", ir);
        
        // Basic verification
        assert!(ir.contains("define i64 @add_const(i64"));
        assert!(ir.contains("add i64"));
    }

    #[test]
    fn test_empty_struct() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Struct(ItemStruct {
                    name: "Empty".to_string(),
                    fields: vec![],
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        println!("Generated IR:\n{}", ir);
        
        // Basic verification
        assert!(ir.contains("%Empty = type {}"));
    }

    #[test]
    fn test_simple_function() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Function(ItemFunction {
                    name: "add42".to_string(),
                    params: vec![
                        FunctionParam {
                            name: "x".to_string(),
                            ty: Type::Int,
                        },
                    ],
                    return_type: Some(Type::Int),
                    body: Block {
                        statements: vec![
                            Stmt::Let {
                                name: "result".to_string(),
                                ty: Some(Type::Int),
                                value: Some(Expr::Binary {
                                    op: BinOp::Add,
                                    lhs: Box::new(Expr::VarRef("x".to_string())),
                                    rhs: Box::new(Expr::Literal(Literal::Int(42))),
                                }),
                            },
                            Stmt::Expr(Expr::VarRef("result".to_string())),
                        ],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        
        // Basic verification
        assert!(ir.contains("define i64 @add42(i64"));
        assert!(ir.contains("add i64"));
        assert!(ir.contains(", 42"));
    }

    #[test]
    fn test_if_expression() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Function(ItemFunction {
                    name: "test_if".to_string(),
                    params: vec![
                        FunctionParam {
                            name: "x".to_string(),
                            ty: Type::Int,
                        },
                    ],
                    return_type: Some(Type::Int),
                    body: Block {
                        statements: vec![
                            Stmt::Expr(Expr::If {
                                condition: Box::new(Expr::Binary {
                                    op: BinOp::Eq,
                                    lhs: Box::new(Expr::VarRef("x".to_string())),
                                    rhs: Box::new(Expr::Literal(Literal::Int(0))),
                                }),
                                then_branch: Block {
                                    statements: vec![
                                        Stmt::Expr(Expr::Literal(Literal::Int(42))),
                                    ],
                                },
                                else_branch: Some(Block {
                                    statements: vec![
                                        Stmt::Expr(Expr::Literal(Literal::Int(24))),
                                    ],
                                }),
                            }),
                        ],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        
        // Basic verification
        assert!(ir.contains("define i64 @test_if(i64"));
        assert!(ir.contains("icmp eq"));
        assert!(ir.contains("br i1"));
        assert!(ir.contains("phi i64"));
    }

    #[test]
    fn test_struct_definition() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Struct(ItemStruct {
                    name: "Point".to_string(),
                    fields: vec![
                        StructField {
                            name: "x".to_string(),
                            ty: Type::Int,
                        },
                        StructField {
                            name: "y".to_string(),
                            ty: Type::Int,
                        },
                    ],
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        
        // Basic verification
        assert!(ir.contains("%Point = type { i64, i64 }"));
    }
}
