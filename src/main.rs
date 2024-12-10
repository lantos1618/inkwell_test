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

    fn verify_ir(ir: &str, expected_patterns: &[&str]) {
        println!("Generated IR:\n{}", ir);
        for pattern in expected_patterns {
            assert!(ir.contains(pattern), "Expected IR to contain: {}", pattern);
        }
    }

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
                            Stmt::Expr(Expr::VarRef("x".to_string())),
                        ],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "define i64 @test_var(i64",
            "load i64, ptr %x",
        ]);
    }

    #[test]
    fn test_nested_struct() {
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
                Item::Struct(ItemStruct {
                    name: "Rectangle".to_string(),
                    fields: vec![
                        StructField {
                            name: "top_left".to_string(),
                            ty: Type::Struct("Point".to_string()),
                        },
                        StructField {
                            name: "bottom_right".to_string(),
                            ty: Type::Struct("Point".to_string()),
                        },
                    ],
                }),
                Item::Function(ItemFunction {
                    name: "create_rect".to_string(),
                    params: vec![],
                    return_type: Some(Type::Struct("Rectangle".to_string())),
                    body: Block {
                        statements: vec![],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "%Point = type { i64, i64 }",
            "%Rectangle = type { %Point, %Point }",
            "define %Rectangle @create_rect()",
        ]);
    }

    #[test]
    fn test_array_type() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Struct(ItemStruct {
                    name: "IntArray".to_string(),
                    fields: vec![
                        StructField {
                            name: "data".to_string(),
                            ty: Type::Array(Box::new(Type::Int)),
                        },
                    ],
                }),
                Item::Function(ItemFunction {
                    name: "create_array".to_string(),
                    params: vec![],
                    return_type: Some(Type::Struct("IntArray".to_string())),
                    body: Block {
                        statements: vec![],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "%IntArray = type { [0 x i64] }",
            "define %IntArray @create_array()",
        ]);
    }

    #[test]
    fn test_function_types() {
        let context = Context::create();
        let mut codegen_ctx = CodegenContext::new(&context, "test_module");

        let program = Program {
            items: vec![
                Item::Struct(ItemStruct {
                    name: "Callback".to_string(),
                    fields: vec![
                        StructField {
                            name: "func".to_string(),
                            ty: Type::Function {
                                params: vec![Type::Int],
                                return_type: Box::new(Type::Int),
                            },
                        },
                    ],
                }),
                Item::Function(ItemFunction {
                    name: "create_callback".to_string(),
                    params: vec![],
                    return_type: Some(Type::Struct("Callback".to_string())),
                    body: Block {
                        statements: vec![],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "%Callback = type { ptr }",
            "define %Callback @create_callback()",
        ]);
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
                Item::Function(ItemFunction {
                    name: "create_empty".to_string(),
                    params: vec![],
                    return_type: Some(Type::Struct("Empty".to_string())),
                    body: Block {
                        statements: vec![],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "%Empty = type {}",
            "define %Empty @create_empty()",
        ]);
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
                Item::Function(ItemFunction {
                    name: "create_point".to_string(),
                    params: vec![],
                    return_type: Some(Type::Struct("Point".to_string())),
                    body: Block {
                        statements: vec![],
                    },
                }),
            ],
        };

        program.codegen(&mut codegen_ctx);
        let ir = codegen_ctx.module.print_to_string().to_string();
        verify_ir(&ir, &[
            "%Point = type { i64, i64 }",
            "define %Point @create_point()",
        ]);
    }
}
