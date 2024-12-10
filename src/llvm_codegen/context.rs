use inkwell::{
    context::Context,
    module::Module,
    builder::Builder,
    values::{BasicValue, BasicValueEnum, PointerValue, FunctionValue},
    types::BasicTypeEnum,
};
use std::collections::HashMap;

use crate::ast::Type;

pub struct CodegenContext<'ctx> {
    pub context: &'ctx Context,
    pub module: Module<'ctx>,
    pub builder: Builder<'ctx>,
    pub scopes: Vec<HashMap<String, PointerValue<'ctx>>>,
    pub type_cache: HashMap<Type, BasicTypeEnum<'ctx>>,
    pub function_cache: HashMap<String, FunctionValue<'ctx>>,
}

impl<'ctx> CodegenContext<'ctx> {
    pub fn new(context: &'ctx Context, module_name: &str) -> Self {
        let module = context.create_module(module_name);
        let builder = context.create_builder();
        Self {
            context,
            module,
            builder,
            scopes: vec![HashMap::new()],
            type_cache: HashMap::new(),
            function_cache: HashMap::new(),
        }
    }

    pub fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub fn insert_variable(&mut self, name: &str, ptr: PointerValue<'ctx>) {
        self.scopes.last_mut().unwrap().insert(name.to_string(), ptr);
    }

    pub fn get_variable(&self, name: &str) -> Option<PointerValue<'ctx>> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(*val);
            }
        }
        None
    }
} 