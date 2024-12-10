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
        items: vec![Item::Function(ItemFunction {
            name: "add42".to_string(),
            params: vec![FunctionParam {
                name: "x".to_string(),
                ty: AstType::Int,
            }],
            return_type: Some(AstType::Int),
            body: Block {
                statements: vec![
                    Stmt::Let {
                        name: "result".to_string(),
                        ty: Some(AstType::Int),
                        value: Some(Expr::Binary {
                            op: BinOp::Add,
                            lhs: Box::new(Expr::VarRef("x".to_string())),
                            rhs: Box::new(Expr::Literal(Literal::Int(42))),
                        }),
                    },
                    Stmt::Return(Some(Expr::VarRef("result".to_string()))),
                ],
            },
        })],
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
        items: vec![Item::Function(ItemFunction {
            name: "test_if".to_string(),
            params: vec![FunctionParam {
                name: "x".to_string(),
                ty: AstType::Int,
            }],
            return_type: Some(AstType::Int),
            body: Block {
                statements: vec![Stmt::Return(Some(Expr::If {
                    condition: Box::new(Expr::Binary {
                        op: BinOp::Eq,
                        lhs: Box::new(Expr::VarRef("x".to_string())),
                        rhs: Box::new(Expr::Literal(Literal::Int(0))),
                    }),
                    then_branch: Block {
                        statements: vec![Stmt::Expr(Expr::Literal(Literal::Int(42)))],
                    },
                    else_branch: Some(Block {
                        statements: vec![Stmt::Expr(Expr::Literal(Literal::Int(24)))],
                    }),
                }))],
            },
        })],
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
                        ty: AstType::Int,
                    },
                    StructField {
                        name: "y".to_string(),
                        ty: AstType::Int,
                    },
                ],
            }),
            Item::Function(ItemFunction {
                name: "make_point".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: AstType::Int,
                    },
                    FunctionParam {
                        name: "y".to_string(),
                        ty: AstType::Int,
                    },
                ],
                return_type: Some(AstType::Struct("Point".to_string())),
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
    println!("Generated IR:\n{}", ir);

    // Basic verification - more flexible with parameter names
    assert!(ir.contains("%Point = type { i64, i64 }"));
    assert!(ir.contains("define %Point @make_point(i64"));
    assert!(ir.contains(", i64"));
}

#[test]
fn test_while_loop() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![Item::Function(ItemFunction {
            name: "sum_to".to_string(),
            params: vec![FunctionParam {
                name: "n".to_string(),
                ty: AstType::Int,
            }],
            return_type: Some(AstType::Int),
            body: Block {
                statements: vec![
                    Stmt::Let {
                        name: "sum".to_string(),
                        ty: Some(AstType::Int),
                        value: Some(Expr::Literal(Literal::Int(0))),
                    },
                    Stmt::Let {
                        name: "i".to_string(),
                        ty: Some(AstType::Int),
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
                                    ty: Some(AstType::Int),
                                    value: Some(Expr::Binary {
                                        op: BinOp::Add,
                                        lhs: Box::new(Expr::VarRef("sum".to_string())),
                                        rhs: Box::new(Expr::VarRef("i".to_string())),
                                    }),
                                },
                                Stmt::Let {
                                    name: "i".to_string(),
                                    ty: Some(AstType::Int),
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
        })],
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
        items: vec![Item::Function(ItemFunction {
            name: "add_float".to_string(),
            params: vec![
                FunctionParam {
                    name: "x".to_string(),
                    ty: AstType::Float,
                },
                FunctionParam {
                    name: "y".to_string(),
                    ty: AstType::Float,
                },
            ],
            return_type: Some(AstType::Float),
            body: Block {
                statements: vec![Stmt::Return(Some(Expr::Binary {
                    op: BinOp::Add,
                    lhs: Box::new(Expr::VarRef("x".to_string())),
                    rhs: Box::new(Expr::VarRef("y".to_string())),
                }))],
            },
        })],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);

    // Basic verification - more flexible with parameter names
    assert!(ir.contains("define double @add_float(double"));
    assert!(ir.contains(", double"));
    assert!(ir.contains("fadd double"));
}

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
        items: vec![Item::Function(ItemFunction {
            name: "test_var".to_string(),
            params: vec![FunctionParam {
                name: "x".to_string(),
                ty: AstType::Int,
            }],
            return_type: Some(AstType::Int),
            body: Block {
                statements: vec![Stmt::Expr(Expr::VarRef("x".to_string()))],
            },
        })],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    verify_ir(&ir, &["define i64 @test_var(i64", "load i64, ptr %x"]);
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
                        ty: AstType::Int,
                    },
                    StructField {
                        name: "y".to_string(),
                        ty: AstType::Int,
                    },
                ],
            }),
            Item::Struct(ItemStruct {
                name: "Rectangle".to_string(),
                fields: vec![
                    StructField {
                        name: "top_left".to_string(),
                        ty: AstType::Struct("Point".to_string()),
                    },
                    StructField {
                        name: "bottom_right".to_string(),
                        ty: AstType::Struct("Point".to_string()),
                    },
                ],
            }),
            Item::Function(ItemFunction {
                name: "create_rect".to_string(),
                params: vec![],
                return_type: Some(AstType::Struct("Rectangle".to_string())),
                body: Block { statements: vec![] },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    verify_ir(
        &ir,
        &[
            "%Point = type { i64, i64 }",
            "%Rectangle = type { %Point, %Point }",
            "define %Rectangle @create_rect()",
        ],
    );
}

#[test]
fn test_array_type() {
    let context = Context::create();
    let mut codegen_ctx = CodegenContext::new(&context, "test_module");

    let program = Program {
        items: vec![
            Item::Struct(ItemStruct {
                name: "IntArray".to_string(),
                fields: vec![StructField {
                    name: "data".to_string(),
                    ty: AstType::Array(Box::new(AstType::Int)),
                }],
            }),
            Item::Function(ItemFunction {
                name: "create_array".to_string(),
                params: vec![],
                return_type: Some(AstType::Struct("IntArray".to_string())),
                body: Block { statements: vec![] },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    verify_ir(
        &ir,
        &[
            "%IntArray = type { [0 x i64] }",
            "define %IntArray @create_array()",
        ],
    );
}

#[test]
fn test_function_types() {
    let context = Context::create();
    let mut codegen_ctx = CodegenContext::new(&context, "test_module");

    let program = Program {
        items: vec![
            Item::Struct(ItemStruct {
                name: "Callback".to_string(),
                fields: vec![StructField {
                    name: "func".to_string(),
                    ty: AstType::Function {
                        params: vec![AstType::Int],
                        return_type: Box::new(AstType::Int),
                    },
                }],
            }),
            Item::Function(ItemFunction {
                name: "create_callback".to_string(),
                params: vec![],
                return_type: Some(AstType::Struct("Callback".to_string())),
                body: Block { statements: vec![] },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    verify_ir(
        &ir,
        &[
            "%Callback = type { ptr }",
            "define %Callback @create_callback()",
        ],
    );
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
                return_type: Some(AstType::Struct("Empty".to_string())),
                body: Block { statements: vec![] },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    verify_ir(&ir, &["%Empty = type {}", "define %Empty @create_empty()"]);
}

#[test]
fn test_nested_scopes() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "test_scopes".to_string(),
                params: vec![],
                return_type: Some(AstType::Int),
                body: Block {
                    statements: vec![
                        Stmt::Let {
                            name: "x".to_string(),
                            ty: Some(AstType::Int),
                            value: Some(Expr::Literal(Literal::Int(1))),
                        },
                        Stmt::Let {
                            name: "y".to_string(),
                            ty: Some(AstType::Int),
                            value: Some(Expr::Literal(Literal::Int(2))),
                        },
                        Stmt::Expr(Expr::If {
                            condition: Box::new(Expr::Literal(Literal::Bool(true))),
                            then_branch: Block {
                                statements: vec![
                                    Stmt::Let {
                                        name: "x".to_string(),  // Shadows outer x
                                        ty: Some(AstType::Int),
                                        value: Some(Expr::Literal(Literal::Int(3))),
                                    },
                                ],
                            },
                            else_branch: Some(Block {
                                statements: vec![],
                            }),
                        }),
                        Stmt::Return(Some(Expr::VarRef("x".to_string()))),  // Should refer to outer x
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    assert!(ir.contains("define i64 @test_scopes()"));
    assert!(ir.contains("alloca i64")); // Variable allocations
    assert!(ir.contains("store i64 1")); // Initial x value
    assert!(ir.contains("store i64 3")); // Inner x value
}

#[test]
fn test_string_literals() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "test_strings".to_string(),
                params: vec![],
                return_type: Some(AstType::String),
                body: Block {
                    statements: vec![
                        Stmt::Return(Some(Expr::Literal(Literal::String("Hello, World!".to_string())))),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    // More flexible string constant check
    assert!(ir.contains("@str = global [14 x i8] c\"Hello, World!\\00\"")); // String constant
    assert!(ir.contains("define ptr @test_strings()")); // Function returning string (pointer)
}

#[test]
fn test_struct_field_access() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Struct(ItemStruct {
                name: "Person".to_string(),
                fields: vec![
                    StructField {
                        name: "age".to_string(),
                        ty: AstType::Int,
                    },
                    StructField {
                        name: "height".to_string(),
                        ty: AstType::Float,
                    },
                ],
            }),
            Item::Function(ItemFunction {
                name: "get_age".to_string(),
                params: vec![
                    FunctionParam {
                        name: "person".to_string(),
                        ty: AstType::Struct("Person".to_string()),
                    },
                ],
                return_type: Some(AstType::Int),
                body: Block {
                    statements: vec![
                        // For now, just return a constant since field access isn't implemented
                        Stmt::Return(Some(Expr::Literal(Literal::Int(0)))),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    // Check struct type and function signature
    assert!(ir.contains("%Person = type { i64, double }")); // Struct definition
    assert!(ir.contains("define i64 @get_age(%Person %person)")); // Function signature
}

#[test]
fn test_array_operations() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "sum_array".to_string(),
                params: vec![
                    FunctionParam {
                        name: "arr".to_string(),
                        ty: AstType::Array(Box::new(AstType::Int)),
                    },
                    FunctionParam {
                        name: "len".to_string(),
                        ty: AstType::Int,
                    },
                ],
                return_type: Some(AstType::Int),
                body: Block {
                    statements: vec![
                        Stmt::Let {
                            name: "sum".to_string(),
                            ty: Some(AstType::Int),
                            value: Some(Expr::Literal(Literal::Int(0))),
                        },
                        Stmt::Return(Some(Expr::VarRef("sum".to_string()))),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    // More flexible array parameter check
    assert!(ir.contains("define i64 @sum_array([0 x i64]")); // Array parameter
    assert!(ir.contains(", i64")); // Length parameter
}

#[test]
fn test_optional_types() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "maybe_add_one".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: AstType::Optional(Box::new(AstType::Int)),
                    },
                ],
                return_type: Some(AstType::Optional(Box::new(AstType::Int))),
                body: Block {
                    statements: vec![
                        Stmt::Return(Some(Expr::VarRef("x".to_string()))),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    // Check for optional type structure
    assert!(ir.contains("define { i64, i1 } @maybe_add_one")); // Function signature
    assert!(ir.contains("alloca { i64, i1 }")); // Local variable allocation
}

#[test]
fn test_match_expression() {
    let context = Context::create();
    let mut codegen_ctx = create_test_context(&context);

    let program = Program {
        items: vec![
            Item::Function(ItemFunction {
                name: "test_match".to_string(),
                params: vec![
                    FunctionParam {
                        name: "x".to_string(),
                        ty: AstType::Int,
                    },
                ],
                return_type: Some(AstType::Int),
                body: Block {
                    statements: vec![
                        Stmt::Return(Some(Expr::Match {
                            expr: Box::new(Expr::VarRef("x".to_string())),
                            arms: vec![
                                MatchArm {
                                    pattern: Pattern::Literal(Literal::Int(0)),
                                    guard: None,
                                    body: Block {
                                        statements: vec![
                                            Stmt::Expr(Expr::Literal(Literal::Int(42))),
                                        ],
                                    },
                                },
                                MatchArm {
                                    pattern: Pattern::Range {
                                        start: Box::new(Literal::Int(1)),
                                        end: Box::new(Literal::Int(10)),
                                        inclusive: true,
                                    },
                                    guard: None,
                                    body: Block {
                                        statements: vec![
                                            Stmt::Expr(Expr::Literal(Literal::Int(24))),
                                        ],
                                    },
                                },
                                MatchArm {
                                    pattern: Pattern::Wildcard,
                                    guard: None,
                                    body: Block {
                                        statements: vec![
                                            Stmt::Expr(Expr::Literal(Literal::Int(0))),
                                        ],
                                    },
                                },
                            ],
                        })),
                    ],
                },
            }),
        ],
    };

    program.codegen(&mut codegen_ctx);
    let ir = codegen_ctx.module.print_to_string().to_string();
    println!("Generated IR:\n{}", ir);
    
    // Check for match-related IR patterns
    assert!(ir.contains("define i64 @test_match(i64 %x)")); // Function signature
    assert!(ir.contains("switch i64")); // Switch instruction for match
    assert!(ir.contains("i64 0")); // Literal pattern
    assert!(ir.contains("i64 42")); // Return value for first arm
    assert!(ir.contains("i64 24")); // Return value for second arm
    assert!(ir.contains("default:")); // Default case for wildcard pattern
}