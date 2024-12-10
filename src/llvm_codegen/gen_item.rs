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
        let param_types: Vec<BasicMetadataTypeEnum<'ctx>> = self.params
            .iter()
            .map(|p| ctx.llvm_type(&p.ty).into())
            .collect();

        let fn_type = if let Some(ret) = &self.return_type {
            let ret_type = ctx.llvm_type(ret);
            ret_type.fn_type(&param_types, false)
        } else {
            ctx.context.void_type().fn_type(&param_types, false)
        };

        let function = ctx.module.add_function(&self.name, fn_type, None);
        ctx.function_cache.insert(self.name.clone(), function);

        let entry_block = ctx.context.append_basic_block(function, "entry");
        ctx.builder.position_at_end(entry_block);

        ctx.push_scope();

        // Allocate and store parameters
        for (i, param) in self.params.iter().enumerate() {
            let arg = function.get_nth_param(i as u32).unwrap();
            let param_type = ctx.llvm_type(&param.ty);
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
                _ => {
                    stmt.codegen(ctx);
                }
            }
        }

        // Handle return value
        if let Some(_) = &self.return_type {
            if let Some(val) = last_expr_value {
                ctx.builder.build_return(Some(&val)).unwrap();
            } else {
                // If no explicit return value, return 0 for numeric types
                let default_val = ctx.context.i64_type().const_zero().as_basic_value_enum();
                ctx.builder.build_return(Some(&default_val)).unwrap();
            }
        } else {
            ctx.builder.build_return(None).unwrap();
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

        // Then collect field types
        let field_types: Vec<_> = self.fields
            .iter()
            .map(|f| ctx.llvm_type(&f.ty))
            .collect();

        // Set the body (even for empty structs)
        struct_type.set_body(&field_types, false);
    }
} 