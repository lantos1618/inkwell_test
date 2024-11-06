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
    
    // Get just the function declaration part from the IR
    let ir = codegen.module.print_to_string().to_string();
    let decl_line = ir.lines()
        .find(|line| line.contains("declare"))
        .unwrap_or("");
    
    assert_eq!(decl_line.trim(), "declare i32 @add(i32, i32)");
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

#[test]
fn test_loop_with_break() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a test function that counts to 10
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "count_to_ten".to_string(),
            params: vec![],
            return_type: Some(AstType::I64),
        },
        body: vec![
            // Initialize i = 0
            Stmt::VarDecl(VarDecl {
                name: "i".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
            }),
            // while(true)
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                body: vec![
                    // if (i > 10) break;
                    Stmt::If(IfStmt {
                        condition: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Gt,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "i".to_string(),
                                type_: AstType::I64,
                            })),
                            right: Box::new(Expr::Literal(Literal::Int(10))),
                        }))),
                        then_branch: vec![Stmt::Break],
                        else_branch: None,
                    }),
                    // i = i + 1
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
            // return i
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Variable(Variable_ {
                    name: "i".to_string(),
                    type_: AstType::I64,
                }))),
            }),
        ],
    };

    // Compile the function
    let result = codegen.compile_stmt(&Stmt::FuncDef(test_func));
    assert_and_dump(result, &codegen);

    // Get and execute the function
    type CountFunc = unsafe extern "C" fn() -> i64;
    let func: JitFunction<CountFunc> = codegen.get_function("count_to_ten").unwrap();
    
    // Run the function and verify the result
    unsafe {
        assert_eq!(func.call(), 11); // i increments to 11 before the break
    }
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

#[test]
fn test_complex_loop_execution() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a simpler version first to debug:
    // int main() {
    //     int sum = 0;
    //     int i = 0;
    //     while(true) {
    //         i = i + 1;
    //         if (i > 5) {
    //             sum = sum + i;
    //             if (sum > 20) break;
    //         }
    //     }
    //     return sum;
    // }
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "complex_loop".to_string(),
            params: vec![],
            return_type: Some(AstType::I64),
        },
        body: vec![
            // Initialize sum = 0
            Stmt::VarDecl(VarDecl {
                name: "sum".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
            }),
            // Initialize i = 0
            Stmt::VarDecl(VarDecl {
                name: "i".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Literal(Literal::Int(0)))),
            }),
            // while(true)
            Stmt::Loop(LoopStmt {
                condition: Box::new(Expr::Literal(Literal::Bool(true))),
                body: vec![
                    // i = i + 1
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
                    // if (i > 5)
                    Stmt::If(IfStmt {
                        condition: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Gt,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "i".to_string(),
                                type_: AstType::I64,
                            })),
                            right: Box::new(Expr::Literal(Literal::Int(5))),
                        }))),
                        then_branch: vec![
                            // sum = sum + i
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
                            // if (sum > 20) break
                            Stmt::If(IfStmt {
                                condition: Box::new(Expr::Binary(Box::new(Binary {
                                    op: BinaryOp::Gt,
                                    left: Box::new(Expr::Variable(Variable_ {
                                        name: "sum".to_string(),
                                        type_: AstType::I64,
                                    })),
                                    right: Box::new(Expr::Literal(Literal::Int(20))),
                                }))),
                                then_branch: vec![Stmt::Break],
                                else_branch: None,
                            }),
                        ],
                        else_branch: None,
                    }),
                ],
            }),
            // return sum
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Variable(Variable_ {
                    name: "sum".to_string(),
                    type_: AstType::I64,
                }))),
            }),
        ],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(test_func));
    assert_and_dump(result, &codegen);

    // Get and execute the function
    type LoopFunc = unsafe extern "C" fn() -> i64;
    let func: JitFunction<LoopFunc> = codegen.get_function("complex_loop").unwrap();
    
    // Run the function and verify the result
    // The loop should add 6 + 7 + 8 = 21, then break
    unsafe {
        assert_eq!(func.call(), 21);
    }
}

#[test]
fn test_nested_if_execution() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a function that implements:
    // int test(int x, int y) {
    //     if (x > 5) {
    //         if (y > 10) {
    //             return 3;
    //         }
    //         return 2;
    //     }
    //     return 1;
    // }
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "nested_if".to_string(),
            params: vec![
                ("x".to_string(), AstType::I64),
                ("y".to_string(), AstType::I64),
            ],
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
                    Stmt::If(IfStmt {
                        condition: Box::new(Expr::Binary(Box::new(Binary {
                            op: BinaryOp::Gt,
                            left: Box::new(Expr::Variable(Variable_ {
                                name: "y".to_string(),
                                type_: AstType::I64,
                            })),
                            right: Box::new(Expr::Literal(Literal::Int(10))),
                        }))),
                        then_branch: vec![
                            Stmt::Return(Return {
                                value: Some(Box::new(Expr::Literal(Literal::Int(3)))),
                            }),
                        ],
                        else_branch: None,
                    }),
                    Stmt::Return(Return {
                        value: Some(Box::new(Expr::Literal(Literal::Int(2)))),
                    }),
                ],
                else_branch: Some(vec![
                    Stmt::Return(Return {
                        value: Some(Box::new(Expr::Literal(Literal::Int(1)))),
                    }),
                ]),
            }),
        ],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(test_func));
    assert_and_dump(result, &codegen);

    type TestFunc = unsafe extern "C" fn(i64, i64) -> i64;
    let func: JitFunction<TestFunc> = codegen.get_function("nested_if").unwrap();
    
    unsafe {
        assert_eq!(func.call(3, 15), 1);  // x <= 5
        assert_eq!(func.call(7, 5), 2);   // x > 5, y <= 10
        assert_eq!(func.call(7, 15), 3);  // x > 5, y > 10
    }
}

#[test]
fn test_arithmetic_execution() {
    let context = Context::create();
    let codegen = setup_codegen(&context);

    // Create a function that implements:
    // int calc(int x, int y) {
    //     int a = x * 2;
    //     int b = y + 5;
    //     return (a + b) * (a - b);
    // }
    let test_func = FuncDef {
        decl: FuncDecl {
            name: "calc".to_string(),
            params: vec![
                ("x".to_string(), AstType::I64),
                ("y".to_string(), AstType::I64),
            ],
            return_type: Some(AstType::I64),
        },
        body: vec![
            // a = x * 2
            Stmt::VarDecl(VarDecl {
                name: "a".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "x".to_string(),
                        type_: AstType::I64,
                    })),
                    right: Box::new(Expr::Literal(Literal::Int(2))),
                })))),
            }),
            // b = y + 5
            Stmt::VarDecl(VarDecl {
                name: "b".to_string(),
                type_: AstType::I64,
                init: Some(Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Add,
                    left: Box::new(Expr::Variable(Variable_ {
                        name: "y".to_string(),
                        type_: AstType::I64,
                    })),
                    right: Box::new(Expr::Literal(Literal::Int(5))),
                })))),
            }),
            // return (a + b) * (a - b)
            Stmt::Return(Return {
                value: Some(Box::new(Expr::Binary(Box::new(Binary {
                    op: BinaryOp::Mul,
                    left: Box::new(Expr::Binary(Box::new(Binary {
                        op: BinaryOp::Add,
                        left: Box::new(Expr::Variable(Variable_ {
                            name: "a".to_string(),
                            type_: AstType::I64,
                        })),
                        right: Box::new(Expr::Variable(Variable_ {
                            name: "b".to_string(),
                            type_: AstType::I64,
                        })),
                    }))),
                    right: Box::new(Expr::Binary(Box::new(Binary {
                        op: BinaryOp::Sub,
                        left: Box::new(Expr::Variable(Variable_ {
                            name: "a".to_string(),
                            type_: AstType::I64,
                        })),
                        right: Box::new(Expr::Variable(Variable_ {
                            name: "b".to_string(),
                            type_: AstType::I64,
                        })),
                    }))),
                })))),
            }),
        ],
    };

    let result = codegen.compile_stmt(&Stmt::FuncDef(test_func));
    assert_and_dump(result, &codegen);

    type CalcFunc = unsafe extern "C" fn(i64, i64) -> i64;
    let func: JitFunction<CalcFunc> = codegen.get_function("calc").unwrap();
    
    unsafe {
        // For x=3, y=2:
        // a = 3 * 2 = 6
        // b = 2 + 5 = 7
        // result = (6 + 7) * (6 - 7) = 13 * -1 = -13
        assert_eq!(func.call(3, 2), -13);
        
        // For x=5, y=3:
        // a = 5 * 2 = 10
        // b = 3 + 5 = 8
        // result = (10 + 8) * (10 - 8) = 18 * 2 = 36
        assert_eq!(func.call(5, 3), 36);
    }
}
