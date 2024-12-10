use super::CodegenContext;
use crate::ast::Type;
use inkwell::{
    types::{BasicType, BasicTypeEnum, BasicMetadataTypeEnum},
    AddressSpace,
};

impl<'ctx> CodegenContext<'ctx> {
    pub fn llvm_type(&mut self, ty: &Type) -> BasicTypeEnum<'ctx> {
        if let Some(cached) = self.type_cache.get(ty) {
            return *cached;
        }

        let llvm_ty = match ty {
            Type::Int => self.context.i64_type().into(),
            Type::Float => self.context.f64_type().into(),
            Type::Bool => self.context.bool_type().into(),
            Type::String => self.context.i8_type().ptr_type(AddressSpace::default()).into(),
            Type::Char => self.context.i8_type().into(),
            Type::Array(elem_ty) => {
                let elem_type = self.llvm_type(elem_ty);
                elem_type.array_type(0).into()
            }
            Type::Struct(name) => {
                // First, create an opaque struct type to handle recursive types
                let struct_type = self.context.opaque_struct_type(name);
                // Store it in the cache immediately to handle recursive types
                self.type_cache.insert(ty.clone(), struct_type.into());
                struct_type.into()
            }
            Type::Function { params, return_type } => {
                let param_types: Vec<BasicMetadataTypeEnum> = params.iter()
                    .map(|p| self.llvm_type(p).into())
                    .collect();
                let ret_type = self.llvm_type(return_type);
                ret_type.fn_type(&param_types, false).ptr_type(AddressSpace::default()).into()
            }
            Type::Reference(inner) => {
                self.llvm_type(inner).ptr_type(AddressSpace::default()).into()
            }
            Type::Optional(inner) => {
                // Optional is represented as a struct { value: T, is_some: bool }
                let inner_type = self.llvm_type(inner);
                let struct_type = self.context.struct_type(
                    &[inner_type, self.context.bool_type().into()],
                    false
                );
                struct_type.into()
            }
        };

        // Only insert into cache if not already inserted (for recursive types)
        if !self.type_cache.contains_key(ty) {
            self.type_cache.insert(ty.clone(), llvm_ty);
        }
        llvm_ty
    }

    pub fn set_struct_body(&mut self, name: &str, field_types: &[BasicTypeEnum<'ctx>]) {
        if let Some(cached) = self.type_cache.get(&Type::Struct(name.to_string())) {
            if let BasicTypeEnum::StructType(struct_type) = cached {
                struct_type.set_body(field_types, false);
            }
        }
    }
} 