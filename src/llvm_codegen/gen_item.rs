use super::{Codegen, CodegenContext};
use crate::ast::{Item, ItemFunction, ItemStruct, Stmt, Type};
use inkwell::{
    types::{BasicType, BasicTypeEnum, BasicMetadataTypeEnum},
    values::{BasicValue, BasicValueEnum, AnyValue},
};

impl<'ctx> Codegen<'ctx> for Item {
    type Output = ();
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) {
        match self {
            Item::Function(f) => f.codegen(ctx),
            Item::Struct(s) => s.codegen(ctx),
        }
    }
}

impl<'ctx> Codegen<'ctx> for ItemFunction {
    type Output = ();
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) {
        // First, ensure all parameter types are available
        let param_types: Vec<BasicMetadataTypeEnum<'ctx>> = self.params
            .iter()
            .map(|p| {
                match p.ty {
                    Type::Float => ctx.context.f64_type().into(),
                    _ => ctx.llvm_type(&p.ty).into(),
                }
            })
            .collect();

        // Get the return type and create function type
        let fn_type = if let Some(ret) = &self.return_type {
            match ret {
                Type::Float => ctx.context.f64_type().fn_type(&param_types, false),
                Type::Struct(name) => {
                    let struct_type = ctx.llvm_type(ret);
                    struct_type.fn_type(&param_types, false)
                }
                _ => {
                    let ret_type = ctx.llvm_type(ret);
                    ret_type.fn_type(&param_types, false)
                }
            }
        } else {
            ctx.context.void_type().fn_type(&param_types, false)
        };

        let function = ctx.module.add_function(&self.name, fn_type, None);
        
        // Set parameter names
        for (i, param) in self.params.iter().enumerate() {
            function.get_nth_param(i as u32).unwrap().set_name(&param.name);
        }
        
        ctx.function_cache.insert(self.name.clone(), function);

        let entry_block = ctx.context.append_basic_block(function, "entry");
        ctx.builder.position_at_end(entry_block);

        ctx.push_scope();

        // Allocate and store parameters
        for (i, param) in self.params.iter().enumerate() {
            let arg = function.get_nth_param(i as u32).unwrap();
            let param_type = match param.ty {
                Type::Float => ctx.context.f64_type().into(),
                _ => ctx.llvm_type(&param.ty),
            };
            let alloc = ctx.builder.build_alloca(param_type, &param.name).unwrap();
            ctx.builder.build_store(alloc, arg).unwrap();
            ctx.insert_variable(&param.name, alloc, param.ty.clone());
        }

        // Generate code for the function body
        let mut last_expr_value = None;
        for stmt in &self.body.statements {
            match stmt {
                Stmt::Expr(e) => {
                    last_expr_value = Some(e.codegen(ctx));
                }
                Stmt::Return(Some(expr)) => {
                    let val = expr.codegen(ctx);
                    ctx.builder.build_return(Some(&val)).unwrap();
                    break;
                }
                Stmt::Return(None) => {
                    ctx.builder.build_return(None).unwrap();
                    break;
                }
                _ => {
                    stmt.codegen(ctx);
                }
            }
        }

        // Handle implicit return value if no explicit return was found
        if !ctx.builder.get_insert_block().unwrap().get_terminator().is_some() {
            if let Some(ret_type) = &self.return_type {
                if let Some(val) = last_expr_value {
                    ctx.builder.build_return(Some(&val)).unwrap();
                } else {
                    // Create appropriate default value based on return type
                    let default_val = match ret_type {
                        Type::Int => ctx.context.i64_type().const_zero().as_basic_value_enum(),
                        Type::Float => ctx.context.f64_type().const_float(0.0).as_basic_value_enum(),
                        Type::Bool => ctx.context.bool_type().const_zero().as_basic_value_enum(),
                        Type::Struct(_) => {
                            // Create a zero-initialized struct
                            let struct_type = ctx.llvm_type(ret_type);
                            let alloca = ctx.builder.build_alloca(struct_type, "tmp").unwrap();
                            ctx.builder.build_load(struct_type, alloca, "default_struct").unwrap()
                        }
                        _ => panic!("No default return value for type {:?}", ret_type),
                    };
                    ctx.builder.build_return(Some(&default_val)).unwrap();
                }
            } else {
                ctx.builder.build_return(None).unwrap();
            }
        }

        ctx.pop_scope();
    }
}

impl<'ctx> Codegen<'ctx> for ItemStruct {
    type Output = ();
    fn codegen(&self, ctx: &mut CodegenContext<'ctx>) {
        // First, create the struct type and add it to the cache
        let struct_type = ctx.context.opaque_struct_type(&self.name);
        ctx.type_cache.insert(Type::Struct(self.name.clone()), struct_type.into());

        // Then collect field types - this might trigger codegen of nested structs
        let field_types: Vec<_> = self.fields
            .iter()
            .map(|f| {
                // If this is a struct type, ensure it's generated first
                if let Type::Struct(name) = &f.ty {
                    if !ctx.type_cache.contains_key(&f.ty) {
                        panic!("Struct {} not defined before use", name);
                    }
                }
                ctx.llvm_type(&f.ty)
            })
            .collect();

        // Set the body (even for empty structs)
        struct_type.set_body(&field_types, false);
    }
} 