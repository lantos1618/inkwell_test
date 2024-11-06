use crate::ast::*;
use crate::llvm_codegen::CodeGen;
use inkwell::context::Context;
use inkwell::OptimizationLevel;
use inkwell::execution_engine::JitFunction;

// Add this type alias for test functions

fn setup_codegen<'ctx>(context: &'ctx Context) -> CodeGen<'ctx> {
    let module = context.create_module("test");
    let builder = context.create_builder();
    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    CodeGen::new(context, module, builder, execution_engine)
}

#[test]
fn test_variable_declaration() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let var_decl = VarDecl {
        name: "x".to_string(),
        type_: AstType::I32,
        init: Some(Box::new(Expr::Literal(Literal::Int(42)))),
    };

    assert!(codegen.compile_stmt(&Stmt::VarDecl(var_decl)).is_ok());
}

#[test]
fn test_function_declaration() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let func_decl = FuncDecl {
        name: "add".to_string(),
        params: vec![
            ("x".to_string(), AstType::I32),
            ("y".to_string(), AstType::I32),
        ],
        return_type: Some(AstType::I32),
    };

    codegen.compile_stmt(&Stmt::FuncDecl(func_decl)).unwrap();
    
    let expected_ir = "declare i32 @add(i32 %0, i32 %1)";
    let actual_ir = codegen.module.print_to_string().to_string();
    
    if actual_ir != expected_ir {
        println!("Expected IR:\n{}", expected_ir);
        println!("Generated IR:\n{}", actual_ir);
    }
    assert_eq!(actual_ir, expected_ir);
}

#[test]
fn test_function_definition() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let func_def = FuncDef {
        decl: FuncDecl {
            name: "add".to_string(),
            params: vec![
                ("x".to_string(), AstType::I32),
                ("y".to_string(), AstType::I32),
            ],
            return_type: Some(AstType::I32),
        },
        body: vec![Stmt::Return(Return {
            value: Some(Box::new(Expr::Binary(Box::new(Binary {
                op: BinaryOp::Add,
                left: Box::new(Expr::Variable(Variable_ {
                    name: "x".to_string(),
                    type_: AstType::I32,
                })),
                right: Box::new(Expr::Variable(Variable_ {
                    name: "y".to_string(),
                    type_: AstType::I32,
                })),
            })))),
        })],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(func_def));
    assert_and_dump(result, &codegen);
}

#[test]
fn test_if_statement() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let if_stmt = IfStmt {
        condition: Box::new(Expr::Literal(Literal::Bool(true))),
        then_branch: vec![Stmt::VarDecl(VarDecl {
            name: "x".to_string(),
            type_: AstType::I32,
            init: Some(Box::new(Expr::Literal(Literal::Int(1)))),
        })],
        else_branch: Some(vec![Stmt::VarDecl(VarDecl {
            name: "x".to_string(),
            type_: AstType::I32,
            init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
        })]),
    };

    assert!(codegen.compile_stmt(&Stmt::If(if_stmt)).is_ok());
}

#[test]
fn test_loop_statement() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let loop_stmt = LoopStmt {
        condition: Box::new(Expr::Literal(Literal::Bool(true))),
        body: vec![Stmt::Break],
    };

    assert!(codegen.compile_stmt(&Stmt::Loop(loop_stmt)).is_ok());
}

#[test]
fn test_break_outside_loop() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    assert!(codegen.compile_stmt(&Stmt::Break).is_err());
}

#[test]
fn test_continue_outside_loop() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    assert!(codegen.compile_stmt(&Stmt::Continue).is_err());
}

#[test]
fn test_binary_operations() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let binary_expr = Expr::Binary(Box::new(Binary {
        op: BinaryOp::Add,
        left: Box::new(Expr::Literal(Literal::Int(1))),
        right: Box::new(Expr::Literal(Literal::Int(2))),
    }));

    assert!(codegen.compile_expr(&binary_expr).is_ok());
}

#[test]
fn test_unary_operations() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let unary_expr = Expr::Unary(Box::new(Unary {
        op: UnaryOp::Neg,
        expr: Box::new(Expr::Literal(Literal::Int(1))),
    }));

    assert!(codegen.compile_expr(&unary_expr).is_ok());
}

#[test]
fn test_struct_declaration() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let struct_decl = StructDecl {
        name: "Point".to_string(),
        fields: vec![
            ("x".to_string(), AstType::I32),
            ("y".to_string(), AstType::I32),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::StructDecl(struct_decl)).is_ok());
}

#[test]
fn test_enum_declaration() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let enum_decl = EnumDecl {
        name: "Option".to_string(),
        variants: vec![
            ("None".to_string(), None),
            ("Some".to_string(), Some(AstType::I32)),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::EnumDecl(enum_decl)).is_ok());
}

#[test]
fn test_function_call() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // First declare the function
    let func_decl = FuncDecl {
        name: "add".to_string(),
        params: vec![
            ("x".to_string(), AstType::I32),
            ("y".to_string(), AstType::I32),
        ],
        return_type: Some(AstType::I32),
    };
    codegen.compile_stmt(&Stmt::FuncDecl(func_decl)).unwrap();

    // Then test calling it
    let func_call = FuncCall {
        name: "add".to_string(),
        args: vec![
            Expr::Literal(Literal::Int(1)),
            Expr::Literal(Literal::Int(2)),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::FuncCall(func_call)).is_ok());
}

#[test]
fn test_jit_sum_function() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let sum = codegen
        .jit_compile_sum()
        .expect("Failed to compile sum function");

    unsafe {
        assert_eq!(sum.call(1, 2, 3), 6);
        assert_eq!(sum.call(10, 20, 30), 60);
    }
}

#[test]
fn test_complete_program() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let program = Program {
        statements: vec![
            // Function declaration
            Stmt::FuncDecl(FuncDecl {
                name: "main".to_string(),
                params: vec![],
                return_type: Some(AstType::I32),
            }),
            // Function definition
            Stmt::FuncDef(FuncDef {
                decl: FuncDecl {
                    name: "main".to_string(),
                    params: vec![],
                    return_type: Some(AstType::I32),
                },
                body: vec![
                    Stmt::VarDecl(VarDecl {
                        name: "result".to_string(),
                        type_: AstType::I32,
                        init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
                    }),
                    Stmt::Return(Return {
                        value: Some(Box::new(Expr::Variable(Variable_ {
                            name: "result".to_string(),
                            type_: AstType::I32,
                        }))),
                    }),
                ],
            }),
        ],
    };

    assert!(codegen.compile(&program).is_ok());
}

#[test]
fn test_nested_loops() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let nested_loops = LoopStmt {
        condition: Box::new(Expr::Literal(Literal::Bool(true))),
        body: vec![
            // Inner loop
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                body: vec![
                    // Break from inner loop
                    Stmt::If(IfStmt {
                        condition: Box::new(Expr::Literal(Literal::Bool(true))),
                        then_branch: vec![Stmt::Break],
                        else_branch: None,
                    }),
                ],
            }),
            // Break from outer loop
            Stmt::If(IfStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                then_branch: vec![Stmt::Break],
                else_branch: None,
            }),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::Loop(nested_loops)).is_ok());
}

#[test]
fn test_loop_with_continue() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let loop_with_continue = LoopStmt {
        condition: Box::new(Expr::Literal(Literal::Bool(true))),
        body: vec![
            Stmt::If(IfStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                then_branch: vec![Stmt::Continue],
                else_branch: None,
            }),
            // This should be unreachable after continue
            Stmt::VarDecl(VarDecl {
                name: "x".to_string(),
                type_: AstType::I32,
                init: Some(Box::new(Expr::Literal(Literal::Int(1)))),
            }),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::Loop(loop_with_continue)).is_ok());
}

#[test]
fn test_nested_loops_with_continue() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let nested_loops = LoopStmt {
        condition: Box::new(Expr::Literal(Literal::Bool(true))),
        body: vec![
            // Inner loop
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                body: vec![
                    // Continue inner loop
                    Stmt::If(IfStmt {
                        condition: Box::new(Expr::Literal(Literal::Bool(true))),
                        then_branch: vec![Stmt::Continue],
                        else_branch: None,
                    }),
                    // This should be unreachable
                    Stmt::Break,
                ],
            }),
            // Continue outer loop
            Stmt::If(IfStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                then_branch: vec![Stmt::Continue],
                else_branch: None,
            }),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::Loop(nested_loops)).is_ok());
}

#[test]
fn test_complex_function_definition() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let func_def = FuncDef {
        decl: FuncDecl {
            name: "complex_func".to_string(),
            params: vec![
                ("x".to_string(), AstType::I32),
                ("y".to_string(), AstType::I32),
            ],
            return_type: Some(AstType::I32),
        },
        body: vec![
            // Local variable declaration
            Stmt::VarDecl(VarDecl {
                name: "result".to_string(),
                type_: AstType::I32,
                init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
            }),
            // Loop that modifies the result
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Lt,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "result".to_string(),
                        type_: AstType::I32,
                    })),
                    right: Box::new(Expr::Variable(Variable_ {
                        name: "x".to_string(),
                        type_: AstType::I32,
                    })),
                }))),
                body: vec![
                    // result = result + y
                    Stmt::Assign(Assign {
                        target: Variable_ {
                            name: "result".to_string(),
                            type_: AstType::I32,
                        },
                        value: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Add,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "result".to_string(),
                                type_: AstType::I32,
                            })),
                            right: Box::new(Expr::Variable(Variable_ {
                                name: "y".to_string(),
                                type_: AstType::I32,
                            })),
                        }))),
                    }),
                ],
            }),
            // Return the result
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Variable(Variable_ {
                    name: "result".to_string(),
                    type_: AstType::I32,
                }))),
            }),
        ],
    };

    assert!(codegen.compile_stmt(&Stmt::FuncDef(func_def)).is_ok());
}

#[test]
fn test_variable_declaration_and_use() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a test function that declares a variable and returns its value
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "test_var".to_string(),
            params: vec![],
            return_type: Some(AstType::I64),
        },
        body: vec![
            Stmt::VarDecl(VarDecl {
                name: "x".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(42)))),
            }),
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Variable(Variable_ {
                    name: "x".to_string(),
                    type_: AstType::I64,
                }))),
            }),
        ],
    };

    // Compile the function
    codegen.compile_stmt(&Stmt::FuncDef(test_func)).unwrap();

    // Get and execute the function
    type TestFunc = unsafe extern "C" fn() -> i64;
    let func: JitFunction<TestFunc> = codegen.get_function("test_var").unwrap();
    unsafe {
        assert_eq!(func.call(), 42);
    }
}

#[test]
fn test_function_with_params() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a test function that adds its parameters
    let add_func = FuncDef {
        decl: FuncDecl {
            name: "add".to_string(),
            params: vec![
                ("x".to_string(), AstType::I64),
                ("y".to_string(), AstType::I64),
            ],
            return_type: Some(AstType::I64),
        },
        body: vec![
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "x".to_string(),
                        type_: AstType::I64,
                    })),
                    right: Box::new(Expr::Variable(Variable_ {
                        name: "y".to_string(),
                        type_: AstType::I64,
                    })),
                })))),
            }),
        ],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(add_func));
    assert_and_dump(result, &codegen);

    // Get and execute the function
    type AddFunc = unsafe extern "C" fn(i64, i64) -> i64;
    let func: JitFunction<AddFunc> = codegen.get_function("add").unwrap();
    unsafe {
        assert_eq!(func.call(3, 4), 7);
        assert_eq!(func.call(10, 20), 30);
    }
}

#[test]
fn test_if_statement_execution() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a test function that uses an if statement
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "test_if".to_string(),
            params: vec![("x".to_string(), AstType::I64)],
            return_type: Some(AstType::I64),
        },
        body: vec![
            Stmt::If(IfStmt {
                condition: Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Gt,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "x".to_string(),
                        type_: AstType::I64,
                    })),
                    right: Box::new(Expr::Literal(Literal::Int(5))),
                }))),
                then_branch: vec![
                    Stmt::Return(Return {
                        value: Some(Box::new(Expr::Literal(Literal::Int(1)))),
                    }),
                ],
                else_branch: Some(vec![
                    Stmt::Return(Return {
                        value: Some(Box::new(Expr::Literal(Literal::Int(0)))),
                    }),
                ]),
            }),
        ],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(test_func));
    assert_and_dump(result, &codegen);

    // Get and execute the function
    type IfTestFunc = unsafe extern "C" fn(i64) -> i64;
    let func: JitFunction<IfTestFunc> = codegen.get_function("test_if").unwrap();
    unsafe {
        assert_eq!(func.call(10), 1);  // x > 5, returns 1
        assert_eq!(func.call(3), 0);   // x <= 5, returns 0
    }
}

#[test]
fn test_loop_execution() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a test function that sums numbers from 1 to n
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "sum_to_n".to_string(),
            params: vec![("n".to_string(), AstType::I64)],
            return_type: Some(AstType::I64),
        },
        body: vec![
            // Initialize sum = 0
            Stmt::VarDecl(VarDecl {
                name: "sum".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
            }),
            // Initialize i = 1
            Stmt::VarDecl(VarDecl {
                name: "i".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(1)))),
            }),
            // While i <= n
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Le,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "i".to_string(),
                        type_: AstType::I64,
                    })),
                    right: Box::new(Expr::Variable(Variable_ {
                        name: "n".to_string(),
                        type_: AstType::I64,
                    })),
                }))),
                body: vec![
                    // sum += i
                    Stmt::Assign(Assign {
                        target: Variable_ {
                            name: "sum".to_string(),
                            type_: AstType::I64,
                        },
                        value: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Add,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "sum".to_string(),
                                type_: AstType::I64,
                            })),
                            right: Box::new(Expr::Variable(Variable_ {
                                name: "i".to_string(),
                                type_: AstType::I64,
                            })),
                        }))),
                    }),
                    // i += 1
                    Stmt::Assign(Assign {
                        target: Variable_ {
                            name: "i".to_string(),
                            type_: AstType::I64,
                        },
                        value: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Add,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "i".to_string(),
                                type_: AstType::I64,
                            })),
                            right: Box::new(Expr::Literal(Literal::Int(1))),
                        }))),
                    }),
                ],
            }),
            // Return sum
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Variable(Variable_ {
                    name: "sum".to_string(),
                    type_: AstType::I64,
                }))),
            }),
        ],
    };

    // Compile the function
    codegen.compile_stmt(&Stmt::FuncDef(test_func)).unwrap();

    // Get and execute the function
    type SumFunc = unsafe extern "C" fn(i64) -> i64;
    let func: JitFunction<SumFunc> = codegen.get_function("sum_to_n").unwrap();
    unsafe {
        assert_eq!(func.call(1), 1);     // 1
        assert_eq!(func.call(3), 6);     // 1 + 2 + 3
        assert_eq!(func.call(5), 15);    // 1 + 2 + 3 + 4 + 5
    }
}

#[test]
fn test_string_literal() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let string_expr = Expr::Literal(Literal::String("Hello, World!".to_string()));
    
    // Test that we can compile a string literal
    assert!(codegen.compile_expr(&string_expr).is_ok());
}

#[test]
fn test_string_variable() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    let var_decl = VarDecl {
        name: "message".to_string(),
        type_: AstType::String,
        init: Some(Box::new(Expr::Literal(Literal::String("Hello".to_string())))),
    };

    let result = codegen.compile_stmt(&Stmt::VarDecl(var_decl));
    assert_and_dump(result, &codegen);
}

#[test]
fn test_struct_definition_and_instantiation() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // First declare the struct
    let struct_decl = StructDecl {
        name: "Point".to_string(),
        fields: vec![
            ("x".to_string(), AstType::I64),
            ("y".to_string(), AstType::I64),
        ],
    };
    let result1 = codegen.compile_stmt(&Stmt::StructDecl(struct_decl));
    assert_and_dump(result1, &codegen);

    // Then create an instance
    let struct_def = StructDef {
        name: "Point".to_string(),
        fields: vec![
            ("x".to_string(), Expr::Literal(Literal::Int(10))),
            ("y".to_string(), Expr::Literal(Literal::Int(20))),
        ],
    };
    let result2 = codegen.compile_stmt(&Stmt::StructDef(struct_def));
    assert_and_dump(result2, &codegen);
}

// Helper function for running tests with IR dump on error
fn assert_and_dump<E>(result: Result<(), E>, codegen: &CodeGen) 
where 
    E: std::fmt::Debug
{
    if let Err(e) = &result {
        println!("Error occurred: {:?}", e);
        println!("Generated IR:\n{}", codegen.dump_module());
    }
    assert!(result.is_ok());
}
