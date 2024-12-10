use inkwell::context::Context;
use inkwell_test::{
    ast::*,
    llvm_codegen::{Codegen, CodegenContext},
};

fn create_test_context<'ctx>(context: &'ctx Context) -> CodegenContext<'ctx> {
    CodegenContext::new(context, "test_module")
}

#[test]
fn test_simple_function() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

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
                        Stmt::Return(Some(Expr::VarRef("result".to_string()))),
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
    let mut codegen_ctx = create_test_context(&context);

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
                        Stmt::Return(Some(Expr::If {
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
                        })),
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
    let mut codegen_ctx = create_test_context(&context);

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
                name: "make_point".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: Type::Int,
                    },
                    FunctionParam {
                        name: "y".to_string(),
                        ty: Type::Int,
                    },
                ],
                return_type: Some(Type::Struct("Point".to_string())),
                body: Block {
                    statements: vec![
                        // TODO: Add struct initialization once implemented
                        Stmt::Return(Some(Expr::Literal(Literal::Int(0)))), // Placeholder
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    
    // Basic verification
    assert!(ir.contains("%Point = type { i64, i64 }"));
    assert!(ir.contains("define %Point @make_point(i64, i64)"));
}

#[test]
fn test_while_loop() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "sum_to".to_string(),
                params: vec![
                    FunctionParam {
                        name: "n".to_string(),
                        ty: Type::Int,
                    },
                ],
                return_type: Some(Type::Int),
                body: Block {
                    statements: vec![
                        Stmt::Let {
                            name: "sum".to_string(),
                            ty: Some(Type::Int),
                            value: Some(Expr::Literal(Literal::Int(0))),
                        },
                        Stmt::Let {
                            name: "i".to_string(),
                            ty: Some(Type::Int),
                            value: Some(Expr::Literal(Literal::Int(0))),
                        },
                        Stmt::While {
                            condition: Expr::Binary {
                                op: BinOp::Lt,
                                lhs: Box::new(Expr::VarRef("i".to_string())),
                                rhs: Box::new(Expr::VarRef("n".to_string())),
                            },
                            body: Block {
                                statements: vec![
                                    Stmt::Let {
                                        name: "sum".to_string(),
                                        ty: Some(Type::Int),
                                        value: Some(Expr::Binary {
                                            op: BinOp::Add,
                                            lhs: Box::new(Expr::VarRef("sum".to_string())),
                                            rhs: Box::new(Expr::VarRef("i".to_string())),
                                        }),
                                    },
                                    Stmt::Let {
                                        name: "i".to_string(),
                                        ty: Some(Type::Int),
                                        value: Some(Expr::Binary {
                                            op: BinOp::Add,
                                            lhs: Box::new(Expr::VarRef("i".to_string())),
                                            rhs: Box::new(Expr::Literal(Literal::Int(1))),
                                        }),
                                    },
                                ],
                            },
                        },
                        Stmt::Return(Some(Expr::VarRef("sum".to_string()))),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    
    // Basic verification
    assert!(ir.contains("define i64 @sum_to(i64"));
    assert!(ir.contains("while.cond:"));
    assert!(ir.contains("while.body:"));
    assert!(ir.contains("while.end:"));
    assert!(ir.contains("icmp slt"));
    assert!(ir.contains("br i1"));
}

#[test]
fn test_float_operations() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "add_float".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: Type::Float,
                    },
                    FunctionParam {
                        name: "y".to_string(),
                        ty: Type::Float,
                    },
                ],
                return_type: Some(Type::Float),
                body: Block {
                    statements: vec![
                        Stmt::Return(Some(Expr::Binary {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::VarRef("x".to_string())),
                            rhs: Box::new(Expr::VarRef("y".to_string())),
                        })),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    
    // Basic verification
    assert!(ir.contains("define double @add_float(double, double)"));
    assert!(ir.contains("fadd double"));
} 