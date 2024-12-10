use super::CodegenContext;
use crate::ast::AstType;
use inkwell::{
    types::{BasicType, BasicTypeEnum, BasicMetadataTypeEnum},
    AddressSpace,
};

impl<'ctx> CodegenContext<'ctx> {
    pub fn llvm_type(&mut self, ty: &AstType) -> BasicTypeEnum<'ctx> {
        if let Some(cached) = self.type_cache.get(ty) {
            return *cached;
        }

        let llvm_ty = match ty {
            AstType::Int => self.context.i64_type().into(),
            AstType::Float => self.context.f64_type().into(),
            AstType::Bool => self.context.bool_type().into(),
            AstType::String => self.context.i8_type().ptr_type(AddressSpace::default()).into(),
            AstType::Char => self.context.i8_type().into(),
            AstType::Array(elem_ty) => {
                let elem_type = self.llvm_type(elem_ty);
                elem_type.array_type(0).into()
            }
            AstType::Struct(name) => {
                // Create an opaque struct type
                let struct_type = self.context.opaque_struct_type(name);
                struct_type.into()
            }
            AstType::Function { params, return_type } => {
                let param_types: Vec<BasicMetadataTypeEnum> = params.iter()
                    .map(|p| self.llvm_type(p).into())
                    .collect();
                let ret_type = self.llvm_type(return_type);
                ret_type.fn_type(&param_types, false).ptr_type(AddressSpace::default()).into()
            }
            AstType::Reference(inner) => {
                self.llvm_type(inner).ptr_type(AddressSpace::default()).into()
            }
            AstType::Optional(inner) => {
                let inner_type = self.llvm_type(inner);
                let struct_type = self.context.struct_type(
                    &[inner_type, self.context.bool_type().into()],
                    false
                );
                struct_type.into()
            }
        };

        self.type_cache.insert(ty.clone(), llvm_ty);
        llvm_ty
    }
} 