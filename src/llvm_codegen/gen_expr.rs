use super::{Codegen, CodegenContext};
use crate::ast::{Expr, Literal, BinOp, UnaryOp, Block, Stmt, AstType};
use inkwell::{
    values::{BasicValue, BasicValueEnum, IntValue, FloatValue, BasicMetadataValueEnum, PointerValue},
    types::{BasicType, AnyTypeEnum, AsTypeRef},
    IntPredicate,
    FloatPredicate,
};

impl<'ctx> Codegen<'ctx> for Block {
    type Output = BasicValueEnum<'ctx>;
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) -> Self::Output {
        ctx.push_scope();
        
        let mut last_value = None;
        for stmt in &self.statements {
            match stmt {
                Stmt::Expr(e) => {
                    last_value = Some(e.codegen(ctx));
                }
                _ => {
                    stmt.codegen(ctx);
                }
            }
        }
        
        ctx.pop_scope();
        last_value.unwrap_or_else(|| ctx.context.i64_type().const_zero().as_basic_value_enum())
    }
}

impl<'ctx> Codegen<'ctx> for Expr {
    type Output = BasicValueEnum<'ctx>;
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) -> Self::Output {
        match self {
            Expr::Literal(lit) => match lit {
                Literal::Int(i) => ctx.context.i64_type()
                    .const_int(*i as u64, true)
                    .as_basic_value_enum(),
                Literal::Float(f) => ctx.context.f64_type()
                    .const_float(*f)
                    .as_basic_value_enum(),
                Literal::Bool(b) => ctx.context.bool_type()
                    .const_int(*b as u64, false)
                    .as_basic_value_enum(),
                Literal::String(s) => {
                    let str_type = ctx.context.i8_type().array_type(s.len() as u32 + 1);
                    let global = ctx.module.add_global(str_type, None, "str");
                    let str_val = ctx.context.const_string(s.as_bytes(), true);
                    global.set_initializer(&str_val);
                    global.as_pointer_value().as_basic_value_enum()
                },
                Literal::Char(c) => ctx.context.i8_type()
                    .const_int(*c as u64, false)
                    .as_basic_value_enum(),
            },
            Expr::VarRef(name) => {
                let (ptr, var_type) = ctx.get_variable(name)
                    .unwrap_or_else(|| panic!("Undefined variable {}", name));
                let var_type = var_type.clone();
                let llvm_type = ctx.llvm_type(&var_type);
                ctx.builder.build_load(llvm_type, ptr, name).unwrap()
            }
            Expr::Binary { op, lhs, rhs } => {
                let l = lhs.codegen(ctx);
                let r = rhs.codegen(ctx);

                match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod => {
                        match (l, r) {
                            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                                match op {
                                    BinOp::Add => ctx.builder.build_int_add(l, r, "tmpadd").unwrap().as_basic_value_enum(),
                                    BinOp::Sub => ctx.builder.build_int_sub(l, r, "tmpsub").unwrap().as_basic_value_enum(),
                                    BinOp::Mul => ctx.builder.build_int_mul(l, r, "tmpmul").unwrap().as_basic_value_enum(),
                                    BinOp::Div => ctx.builder.build_int_signed_div(l, r, "tmpdiv").unwrap().as_basic_value_enum(),
                                    BinOp::Mod => ctx.builder.build_int_signed_rem(l, r, "tmpmod").unwrap().as_basic_value_enum(),
                                    _ => unreachable!(),
                                }
                            },
                            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                                match op {
                                    BinOp::Add => ctx.builder.build_float_add(l, r, "tmpadd").unwrap().as_basic_value_enum(),
                                    BinOp::Sub => ctx.builder.build_float_sub(l, r, "tmpsub").unwrap().as_basic_value_enum(),
                                    BinOp::Mul => ctx.builder.build_float_mul(l, r, "tmpmul").unwrap().as_basic_value_enum(),
                                    BinOp::Div => ctx.builder.build_float_div(l, r, "tmpdiv").unwrap().as_basic_value_enum(),
                                    _ => panic!("Invalid float operation: {:?}", op),
                                }
                            },
                            _ => panic!("Type mismatch in arithmetic operation: {:?} and {:?}", l, r),
                        }
                    },
                    BinOp::Eq | BinOp::Ne | BinOp::Lt | BinOp::Le | BinOp::Gt | BinOp::Ge => {
                        match (l, r) {
                            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                                let pred = match op {
                                    BinOp::Eq => IntPredicate::EQ,
                                    BinOp::Ne => IntPredicate::NE,
                                    BinOp::Lt => IntPredicate::SLT,
                                    BinOp::Le => IntPredicate::SLE,
                                    BinOp::Gt => IntPredicate::SGT,
                                    BinOp::Ge => IntPredicate::SGE,
                                    _ => unreachable!(),
                                };
                                ctx.builder.build_int_compare(pred, l, r, "tmpcmp")
                                    .unwrap()
                                    .as_basic_value_enum()
                            },
                            (BasicValueEnum::FloatValue(l), BasicValueEnum::FloatValue(r)) => {
                                let pred = match op {
                                    BinOp::Eq => FloatPredicate::OEQ,
                                    BinOp::Ne => FloatPredicate::ONE,
                                    BinOp::Lt => FloatPredicate::OLT,
                                    BinOp::Le => FloatPredicate::OLE,
                                    BinOp::Gt => FloatPredicate::OGT,
                                    BinOp::Ge => FloatPredicate::OGE,
                                    _ => unreachable!(),
                                };
                                ctx.builder.build_float_compare(pred, l, r, "tmpcmp")
                                    .unwrap()
                                    .as_basic_value_enum()
                            },
                            _ => panic!("Type mismatch in comparison operation: {:?} and {:?}", l, r),
                        }
                    },
                    BinOp::And | BinOp::Or => {
                        match (l, r) {
                            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                                match op {
                                    BinOp::And => ctx.builder.build_and(l, r, "tmpand").unwrap().as_basic_value_enum(),
                                    BinOp::Or => ctx.builder.build_or(l, r, "tmpor").unwrap().as_basic_value_enum(),
                                    _ => unreachable!(),
                                }
                            },
                            _ => panic!("Type mismatch in logical operation: {:?} and {:?}", l, r),
                        }
                    },
                    BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                        match (l, r) {
                            (BasicValueEnum::IntValue(l), BasicValueEnum::IntValue(r)) => {
                                match op {
                                    BinOp::BitAnd => ctx.builder.build_and(l, r, "tmpand").unwrap().as_basic_value_enum(),
                                    BinOp::BitOr => ctx.builder.build_or(l, r, "tmpor").unwrap().as_basic_value_enum(),
                                    BinOp::BitXor => ctx.builder.build_xor(l, r, "tmpxor").unwrap().as_basic_value_enum(),
                                    BinOp::Shl => ctx.builder.build_left_shift(l, r, "tmpshl").unwrap().as_basic_value_enum(),
                                    BinOp::Shr => ctx.builder.build_right_shift(l, r, true, "tmpshr").unwrap().as_basic_value_enum(),
                                    _ => unreachable!(),
                                }
                            },
                            _ => panic!("Type mismatch in bitwise operation: {:?} and {:?}", l, r),
                        }
                    },
                }
            }
            Expr::Unary { op, expr } => {
                let val = expr.codegen(ctx);
                match op {
                    UnaryOp::Neg => match val {
                        BasicValueEnum::IntValue(v) => ctx.builder.build_int_neg(v, "tmpneg").unwrap().as_basic_value_enum(),
                        BasicValueEnum::FloatValue(v) => ctx.builder.build_float_neg(v, "tmpneg").unwrap().as_basic_value_enum(),
                        _ => panic!("Cannot negate non-numeric value"),
                    },
                    UnaryOp::Not => match val {
                        BasicValueEnum::IntValue(v) => ctx.builder.build_not(v, "tmpnot").unwrap().as_basic_value_enum(),
                        _ => panic!("Cannot apply logical not to non-boolean value"),
                    },
                    UnaryOp::Deref => {
                        if let BasicValueEnum::PointerValue(ptr) = val {
                            ctx.builder.build_load(ptr.get_type(), ptr, "tmpderef").unwrap()
                        } else {
                            panic!("Cannot dereference non-pointer value");
                        }
                    },
                    UnaryOp::Ref => panic!("Reference operator not implemented yet"),
                }
            }
            Expr::If { condition, then_branch, else_branch } => {
                let cond_val = condition.codegen(ctx).into_int_value();
                let func = ctx.builder.get_insert_block().unwrap().get_parent().unwrap();
                
                let then_bb = ctx.context.append_basic_block(func, "then");
                let else_bb = ctx.context.append_basic_block(func, "else");
                let merge_bb = ctx.context.append_basic_block(func, "merge");

                ctx.builder.build_conditional_branch(cond_val, then_bb, else_bb).unwrap();

                // Then block
                ctx.builder.position_at_end(then_bb);
                let then_val = then_branch.codegen(ctx);
                let then_val_type = then_val.get_type();
                ctx.builder.build_unconditional_branch(merge_bb).unwrap();
                let then_bb = ctx.builder.get_insert_block().unwrap();

                // Else block
                ctx.builder.position_at_end(else_bb);
                let else_val = match else_branch {
                    Some(e) => e.codegen(ctx),
                    None => ctx.context.i64_type().const_zero().as_basic_value_enum()
                };
                ctx.builder.build_unconditional_branch(merge_bb).unwrap();
                let else_bb = ctx.builder.get_insert_block().unwrap();

                // Merge block
                ctx.builder.position_at_end(merge_bb);
                let phi = ctx.builder.build_phi(then_val_type, "ifval").unwrap();
                phi.add_incoming(&[(&then_val, then_bb), (&else_val, else_bb)]);
                phi.as_basic_value()
            }
            Expr::Call { func, args } => {
                let function = if let Some(f) = ctx.function_cache.get(func) {
                    *f
                } else {
                    panic!("Undefined function: {}", func);
                };

                // Generate all argument values
                let arg_values: Vec<_> = args.iter()
                    .map(|arg| arg.codegen(ctx))
                    .collect();
                // Convert them to BasicMetadataValueEnum
                let arg_values: Vec<BasicMetadataValueEnum> = arg_values.iter()
                    .map(|val| (*val).into())
                    .collect();

                ctx.builder.build_call(function, &arg_values, "tmpcall")
                    .unwrap()
                    .try_as_basic_value()
                    .left()
                    .unwrap_or_else(|| ctx.context.i64_type().const_zero().as_basic_value_enum())
            }
            Expr::FieldAccess { expr, field } => {
                let val = expr.codegen(ctx);
                if let BasicValueEnum::PointerValue(ptr) = val {
                    // TODO: Look up field index in struct type
                    let field_index = 0;
                    let field_ptr = unsafe {
                        ctx.builder.build_struct_gep(ptr.get_type(), ptr, field_index, "field").unwrap()
                    };
                    ctx.builder.build_load(field_ptr.get_type(), field_ptr, field).unwrap()
                } else {
                    panic!("Cannot access field of non-struct value");
                }
            }
            Expr::Index { array, index } => {
                let array_val = array.codegen(ctx);
                let index_val = index.codegen(ctx).into_int_value();
                if let BasicValueEnum::PointerValue(ptr) = array_val {
                    let indices = &[index_val];
                    let elem_ptr = unsafe {
                        ctx.builder.build_gep(ptr.get_type(), ptr, indices, "arrayidx").unwrap()
                    };
                    ctx.builder.build_load(elem_ptr.get_type(), elem_ptr, "elem").unwrap()
                } else {
                    panic!("Cannot index non-array value");
                }
            }
            Expr::Array(elements) => {
                if elements.is_empty() {
                    panic!("Empty array literals not supported yet");
                }
                let first = elements[0].codegen(ctx);
                let elem_type = first.get_type();
                let array_type = elem_type.array_type(elements.len() as u32);
                let alloc = ctx.builder.build_alloca(array_type, "array").unwrap();
                
                for (i, elem) in elements.iter().enumerate() {
                    let elem_val = elem.codegen(ctx);
                    let indices = &[
                        ctx.context.i32_type().const_int(0, false),
                        ctx.context.i32_type().const_int(i as u64, false),
                    ];
                    let elem_ptr = unsafe {
                        ctx.builder.build_gep(alloc.get_type(), alloc, indices, &format!("elem{}", i)).unwrap()
                    };
                    ctx.builder.build_store(elem_ptr, elem_val).unwrap();
                }
                alloc.as_basic_value_enum()
            }
        }
    }
} 