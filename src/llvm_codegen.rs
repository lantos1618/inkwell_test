use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction, UnsafeFunctionPointer};
use inkwell::module::Module;
use inkwell::types::BasicType;

use std::cell::RefCell;
use std::collections::HashMap;

use anyhow::Result;
use crate::ast::*;
use crate::error::CodegenError;

struct LoopContext<'ctx> {
    condition_block: inkwell::basic_block::BasicBlock<'ctx>,
    end_block: inkwell::basic_block::BasicBlock<'ctx>,
}

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    pub module: Module<'ctx>,
    builder: Builder<'ctx>,
    pub execution_engine: ExecutionEngine<'ctx>,
    loop_stack: RefCell<Vec<LoopContext<'ctx>>>,
    variables: RefCell<HashMap<String, inkwell::values::PointerValue<'ctx>>>,
    struct_types: RefCell<HashMap<String, inkwell::types::StructType<'ctx>>>,
}

impl<'ctx> CodeGen<'ctx> {
    /// Constructs a new instance of `CodeGen`.
    pub fn new(
        context: &'ctx Context,
        module: Module<'ctx>,
        builder: Builder<'ctx>,
        execution_engine: ExecutionEngine<'ctx>,
    ) -> Self {
        Self {
            context,
            module,
            builder,
            execution_engine,
            loop_stack: RefCell::new(Vec::new()),
            variables: RefCell::new(HashMap::new()),
            struct_types: RefCell::new(HashMap::new()),
        }
    }

    /// Compiles the program by iterating over its statements.
    pub fn compile(&self, program: &Program) -> Result<()> {
        program.statements.iter().for_each(|stmt| {
            self.compile_stmt(stmt).unwrap();
        });
        Ok(())
    }

    /// Compiles individual statements by pattern matching.
    pub fn compile_stmt(&self, stmt: &Stmt) -> Result<()> {
        match stmt {
            Stmt::FuncDecl(func_decl) => self.compile_func_decl(func_decl),
            Stmt::FuncDef(func_def) => self.compile_func_def(func_def),
            Stmt::VarDecl(var_decl) => self.compile_var_decl(var_decl),
            Stmt::Assign(assign) => self.compile_assign(assign),
            Stmt::Block(block) => self.compile_block(block),
            Stmt::Loop(loop_stmt) => self.compile_loop(loop_stmt),
            Stmt::If(if_stmt) => self.compile_if(if_stmt),
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            },
            Stmt::Return(ret) => self.compile_return(ret),
            Stmt::Break => self.compile_break(),
            Stmt::Continue => self.compile_continue(),
            Stmt::StructDecl(struct_decl) => self.compile_struct_decl(struct_decl),
            Stmt::StructDef(struct_def) => self.compile_struct_def(struct_def),
            Stmt::EnumDecl(enum_decl) => self.compile_enum_decl(enum_decl),
            Stmt::EnumDef(enum_def) => self.compile_enum_def(enum_def),
            Stmt::TypeAlias(type_alias) => self.compile_type_alias(type_alias),
            Stmt::FuncCall(func_call) => {
                self.compile_func_call(func_call)?;
                Ok(())
            },
        }
    }

    /// Translates AST types to LLVM types.
    fn into_llvm_type(&self, ast_type: &AstType) -> Result<inkwell::types::BasicTypeEnum<'ctx>> {
        match ast_type {
            // Integer types
            AstType::I8 => Ok(self.context.i8_type().into()),
            AstType::I16 => Ok(self.context.i16_type().into()),
            AstType::I32 => Ok(self.context.i32_type().into()),
            AstType::I64 => Ok(self.context.i64_type().into()),
            AstType::U8 => Ok(self.context.i8_type().into()),
            AstType::U16 => Ok(self.context.i16_type().into()),
            AstType::U32 => Ok(self.context.i32_type().into()),
            AstType::U64 => Ok(self.context.i64_type().into()),
            // Floating point types
            AstType::F32 => Ok(self.context.f32_type().into()),
            AstType::F64 => Ok(self.context.f64_type().into()),
            // Other basic types
            AstType::Bool => Ok(self.context.bool_type().into()),
            AstType::Char => Ok(self.context.i8_type().into()),
            // Complex types - currently unsupported
            AstType::Void => Err(CodegenError::UnsupportedType.into()),
            AstType::String => Ok(self.context.ptr_type(inkwell::AddressSpace::default()).into()),
            AstType::Struct(name) => {
                if let Some(struct_type) = self.struct_types.borrow().get(name) {
                    Ok(struct_type.as_basic_type_enum())
                } else {
                    // Create an opaque struct type if it doesn't exist yet
                    let struct_type = self.context.opaque_struct_type(name);
                    self.struct_types.borrow_mut().insert(name.clone(), struct_type);
                    Ok(struct_type.as_basic_type_enum())
                }
            },
            AstType::Enum(_) => Err(CodegenError::UnsupportedType.into()),
            AstType::TypeAlias(_) => Err(CodegenError::UnsupportedType.into()),
        }
    }

    fn compile_func_decl(&self, func_decl: &FuncDecl) -> Result<()> {
        // Convert parameter types to LLVM types
        let param_types: Result<Vec<_>> = func_decl.params
            .iter()
            .map(|(_, ty)| self.into_llvm_type(ty))
            .collect();
        let param_types = param_types?;

        // Convert BasicTypeEnum to BasicMetadataTypeEnum
        let param_types: Vec<_> = param_types.iter()
            .map(|ty| (*ty).into())
            .collect();

        // Create function type
        let fn_type = match &func_decl.return_type {
            Some(ret_type) => {
                let ret_type = self.into_llvm_type(ret_type)?;
                ret_type.fn_type(&param_types, false)
            },
            None => self.context.void_type().fn_type(&param_types, false),
        };

        // Add function to module with proper parameter names
        let function = self.module.add_function(&func_decl.name, fn_type, None);
        
        // Set parameter names
        for (i, (param_name, _)) in func_decl.params.iter().enumerate() {
            function.get_nth_param(i as u32)
                .unwrap()
                .set_name(param_name);
        }

        Ok(())
    }

    fn compile_func_def(&self, func_def: &FuncDef) -> Result<()> {
        // First compile the declaration
        self.compile_func_decl(&func_def.decl)?;

        // Get the function
        let function = self.module.get_function(&func_def.decl.name)
            .ok_or(CodegenError::FunctionNotFound)?;

        // Create entry block
        let entry_block = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry_block);

        // Clear variables map for new function scope
        self.variables.borrow_mut().clear();

        // Create allocas for parameters
        for (i, (param_name, param_type)) in func_def.decl.params.iter().enumerate() {
            let param = function.get_nth_param(i as u32)
                .ok_or(CodegenError::UnsupportedType)?;
            
            let alloca = self.builder.build_alloca(
                self.into_llvm_type(param_type)?,
                param_name
            ).map_err(|_| CodegenError::UnsupportedType)?;
            
            self.builder.build_store(alloca, param)
                .map_err(|_| CodegenError::UnsupportedType)?;
            
            self.variables.borrow_mut().insert(param_name.clone(), alloca);
        }

        // Compile function body
        for stmt in &func_def.body {
            self.compile_stmt(stmt)?;
        }

        // Verify the function
        if function.verify(true) {
            Ok(())
        } else {
            function.print_to_stderr();
            Err(CodegenError::UnsupportedExpression.into())
        }
    }

    fn compile_var_decl(&self, var_decl: &VarDecl) -> Result<()> {
        self.ensure_insertion_block()?;
        let var_type = self.into_llvm_type(&var_decl.type_)?;
        
        // Create alloca instruction at the entry of the function
        let alloca = self.builder.build_alloca(var_type, &var_decl.name)
            .map_err(|_| CodegenError::UnsupportedType)?;
        
        // Store the variable in our map
        self.variables.borrow_mut().insert(var_decl.name.clone(), alloca);
        
        // If there's an initializer, compile it and store the result
        if let Some(init) = &var_decl.init {
            let init_val = self.compile_expr(init)?;
            self.builder.build_store(alloca, init_val)
                .map_err(|_| CodegenError::UnsupportedType)?;
        }
        
        Ok(())
    }

    fn compile_assign(&self, assign: &Assign) -> Result<()> {
        // First check local variables
        if let Some(var_ptr) = self.variables.borrow().get(&assign.target.name) {
            let value = self.compile_expr(&assign.value)?;
            self.builder.build_store(*var_ptr, value)
                .map_err(|_| CodegenError::UnsupportedType)?;
            return Ok(());
        }
        
        // Then check global variables
        let var_ptr = self.module.get_global(&assign.target.name)
            .ok_or(CodegenError::VariableNotFound)?;
        
        let value = self.compile_expr(&assign.value)?;
        self.builder.build_store(var_ptr.as_pointer_value(), value)
            .map_err(|_| CodegenError::UnsupportedType)?;
        Ok(())
    }

    fn compile_block(&self, block: &Block) -> Result<()> {
        for stmt in block {
            self.compile_stmt(stmt)?;
        }
        Ok(())
    }

    pub fn compile_expr(&self, expr: &Expr) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        self.ensure_insertion_block()?;
        match expr {
            Expr::Literal(lit) => self.compile_literal(lit),
            Expr::Variable(var) => self.compile_variable(var),
            Expr::Binary(binary) => self.compile_binary_op(binary),
            Expr::Unary(unary) => self.compile_unary_op(unary),
            Expr::FuncCall(call) => self.compile_func_call(call),
            _ => Err(CodegenError::UnsupportedExpression.into()),
        }
    }

    // Helper methods for expression compilation
    fn compile_literal(&self, lit: &Literal) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        match lit {
            Literal::Int(i) => Ok(self.context.i64_type().const_int(*i as u64, false).into()),
            Literal::Float(f) => Ok(self.context.f64_type().const_float(*f).into()),
            Literal::Bool(b) => Ok(self.context.bool_type().const_int(*b as u64, false).into()),
            Literal::Char(c) => Ok(self.context.i8_type().const_int(*c as u64, false).into()),
            Literal::String(s) => {
                // Create string constant in the current context
                let str_type = self.context.i8_type().array_type(s.len() as u32 + 1);
                let global = self.module.add_global(str_type, None, "str");
                global.set_constant(true);
                global.set_unnamed_addr(true);
                
                // Create string constant with null terminator
                let mut string_chars: Vec<u8> = s.bytes().collect();
                string_chars.push(0); // Null terminator
                let const_string = self.context.i8_type()
                    .const_array(&string_chars.iter()
                        .map(|c| self.context.i8_type().const_int(*c as u64, false))
                        .collect::<Vec<_>>());
                
                global.set_initializer(&const_string);
                
                // Get pointer to the string
                Ok(global.as_pointer_value().into())
            }
        }
    }

    fn compile_variable(&self, var: &Variable_) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        // First check local variables
        if let Some(var_ptr) = self.variables.borrow().get(&var.name) {
            let loaded = self.builder.build_load(
                self.into_llvm_type(&var.type_)?,  // Use the variable's type
                *var_ptr,
                &var.name
            ).map_err(|_| CodegenError::UnsupportedType)?;
            return Ok(loaded);
        }
        
        // Then check global variables
        let var_ptr = self.module.get_global(&var.name)
            .ok_or(CodegenError::VariableNotFound)?;
        
        Ok(self.builder.build_load(
            self.into_llvm_type(&var.type_)?,  // Use the variable's type
            var_ptr.as_pointer_value(),
            &var.name,
        ).map_err(|_| CodegenError::UnsupportedType)?)
    }

    fn compile_func_call(&self, func_call: &FuncCall) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        let function = self.module.get_function(&func_call.name)
            .ok_or(CodegenError::FunctionNotFound)?;
        
        // Compile arguments
        let mut compiled_args = Vec::new();
        for arg in &func_call.args {
            let compiled_arg = self.compile_expr(arg)?;
            compiled_args.push(compiled_arg.into());
        }
        
        Ok(self.builder.build_call(function, &compiled_args, "calltmp")
            .unwrap()
            .try_as_basic_value()
            .left()
            .ok_or(CodegenError::UnsupportedExpression)?)
    }

    fn compile_if(&self, if_stmt: &IfStmt) -> Result<()> {
        self.ensure_insertion_block()?;
        
        let parent = self.builder.get_insert_block()
            .ok_or(CodegenError::InvalidCondition)?
            .get_parent()
            .ok_or(CodegenError::InvalidCondition)?;

        let then_block = self.context.append_basic_block(parent, "then");
        let else_block = self.context.append_basic_block(parent, "else");
        let merge_block = self.context.append_basic_block(parent, "merge");

        // Compile condition
        let condition = self.compile_expr(&if_stmt.condition)?;
        let condition = match condition {
            inkwell::values::BasicValueEnum::IntValue(val) => val,
            _ => return Err(CodegenError::InvalidCondition.into()),
        };

        // Branch to then or else block
        self.builder.build_conditional_branch(condition, then_block, else_block)
            .map_err(|_| CodegenError::InvalidCondition)?;

        // Compile then block
        self.builder.position_at_end(then_block);
        for stmt in &if_stmt.then_branch {
            self.compile_stmt(stmt)?;
        }
        
        let then_terminates = self.builder.get_insert_block()
            .unwrap()
            .get_terminator()
            .is_some();

        if !then_terminates {
            self.builder.build_unconditional_branch(merge_block)
                .map_err(|_| CodegenError::InvalidCondition)?;
        }

        // Compile else block
        self.builder.position_at_end(else_block);
        if let Some(else_branch) = &if_stmt.else_branch {
            for stmt in else_branch {
                self.compile_stmt(stmt)?;
            }
        }
        
        let else_terminates = self.builder.get_insert_block()
            .unwrap()
            .get_terminator()
            .is_some();

        if !else_terminates {
            self.builder.build_unconditional_branch(merge_block)
                .map_err(|_| CodegenError::InvalidCondition)?;
        }

        // Only position at merge block if either branch could reach it
        if !then_terminates || !else_terminates {
            self.builder.position_at_end(merge_block);
        } else {
            // If both branches terminate (e.g., with return statements),
            // we can delete the unreachable merge block
            unsafe {
                merge_block.delete().map_err(|_| CodegenError::InvalidCondition)?;
            }
        }

        Ok(())
    }

    fn compile_loop(&self, loop_stmt: &LoopStmt) -> Result<()> {
        self.ensure_insertion_block()?;
        let parent = self.builder.get_insert_block()
            .ok_or(CodegenError::InvalidCondition)?
            .get_parent()
            .ok_or(CodegenError::InvalidCondition)?;
        
        let cond_block = self.context.append_basic_block(parent, "loop_cond");
        let body_block = self.context.append_basic_block(parent, "loop_body");
        let end_block = self.context.append_basic_block(parent, "loop_end");

        // Push loop context
        self.loop_stack.borrow_mut().push(LoopContext {
            condition_block: cond_block,
            end_block: end_block,
        });

        // Jump to condition
        self.builder.build_unconditional_branch(cond_block)
            .map_err(|_| CodegenError::InvalidCondition)?;

        // Compile condition
        self.builder.position_at_end(cond_block);
        let condition = self.compile_expr(&loop_stmt.condition)?;
        let condition = match condition {
            inkwell::values::BasicValueEnum::IntValue(val) => val,
            _ => return Err(CodegenError::InvalidCondition.into()),
        };

        self.builder.build_conditional_branch(condition, body_block, end_block)
            .map_err(|_| CodegenError::InvalidCondition)?;

        // Compile loop body
        self.builder.position_at_end(body_block);
        for stmt in &loop_stmt.body {
            self.compile_stmt(stmt)?;
        }

        // Only branch back to condition if the block doesn't already have a terminator
        if self.builder.get_insert_block().unwrap().get_terminator().is_none() {
            self.builder.build_unconditional_branch(cond_block)
                .map_err(|_| CodegenError::InvalidCondition)?;
        }

        // Continue after loop
        self.builder.position_at_end(end_block);

        // Pop loop context
        self.loop_stack.borrow_mut().pop();
        
        Ok(())
    }

    fn compile_return(&self, ret: &Return) -> Result<()> {
        match &ret.value {
            Some(expr) => {
                let return_value = self.compile_expr(expr)?;
                self.builder.build_return(Some(&return_value)).unwrap();
            }
            None => {
                self.builder.build_return(None).unwrap();
            }
        }
        Ok(())
    }

    fn compile_break(&self) -> Result<()> {
        let loop_stack = self.loop_stack.borrow();
        let loop_context = loop_stack.last()
            .ok_or(CodegenError::BreakOutsideLoop)?;
        
        // Branch to loop end block
        self.builder.build_unconditional_branch(loop_context.end_block)
            .map_err(|_| CodegenError::InvalidCondition)?;

        // Create unreachable block for any code after break
        let parent = self.builder.get_insert_block().unwrap().get_parent().unwrap();
        let unreachable_block = self.context.append_basic_block(parent, "after_break");
        self.builder.position_at_end(unreachable_block);
        self.builder.build_unreachable().map_err(|_| CodegenError::InvalidCondition)?;
        
        Ok(())
    }

    fn compile_continue(&self) -> Result<()> {
        // Create a longer-lived borrow
        let loop_stack = self.loop_stack.borrow();
        let loop_context = loop_stack.last()
            .ok_or(CodegenError::ContinueOutsideLoop)?;
        
        self.builder.build_unconditional_branch(loop_context.condition_block).unwrap();
        
        // Create a new block for unreachable code after continue
        let parent = self.builder.get_insert_block().unwrap().get_parent().unwrap();
        let unreachable_block = self.context.append_basic_block(parent, "after_continue");
        self.builder.position_at_end(unreachable_block);
        
        Ok(())
    }

    fn compile_struct_decl(&self, struct_decl: &StructDecl) -> Result<()> {
        let field_types: Result<Vec<_>> = struct_decl.fields
            .iter()
            .map(|(_, ty)| self.into_llvm_type(ty))
            .collect();
        let field_types = field_types?;
        
        let struct_type = self.context.opaque_struct_type(&struct_decl.name);
        struct_type.set_body(&field_types, false);
        
        self.struct_types.borrow_mut().insert(struct_decl.name.clone(), struct_type);
        Ok(())
    }

    fn compile_struct_def(&self, struct_def: &StructDef) -> Result<()> {
        let struct_type = self.struct_types.borrow()
            .get(&struct_def.name)
            .ok_or(CodegenError::UnsupportedType)?
            .clone();

        self.ensure_insertion_block()?;

        // Allocate space for the struct
        let struct_ptr = self.builder.build_alloca(struct_type, &struct_def.name)
            .map_err(|_| CodegenError::UnsupportedType)?;

        // Compile and store each field
        for (i, (field_name, field_expr)) in struct_def.fields.iter().enumerate() {
            let field_value = self.compile_expr(field_expr)?;
            
            // Get pointer to the field
            let field_ptr = self.builder.build_struct_gep(
                struct_type,
                struct_ptr,
                i as u32,
                field_name
            ).map_err(|_| CodegenError::UnsupportedType)?;
            
            // Store the field value
            self.builder.build_store(field_ptr, field_value)
                .map_err(|_| CodegenError::UnsupportedType)?;
        }

        // Store the struct pointer in our variables map
        self.variables.borrow_mut().insert(struct_def.name.clone(), struct_ptr);
        Ok(())
    }

    fn compile_enum_decl(&self, enum_decl: &EnumDecl) -> Result<()> {
        // Create a discriminated union type for the enum
        // This is a simplified version - real enums need more complex handling
        let discriminant_type = self.context.i32_type();
        let mut variant_types = vec![discriminant_type.into()];
        
        for (_, variant_type) in &enum_decl.variants {
            if let Some(ty) = variant_type {
                variant_types.push(self.into_llvm_type(ty)?);
            }
        }
        
        self.context.struct_type(&variant_types, false);
        Ok(())
    }

    fn compile_enum_def(&self, _enum_def: &EnumDef) -> Result<()> {
        // This would create an instance of an enum variant
        // You'll need to maintain a mapping of enum types
        Err(CodegenError::Unimplemented.into())
    }

    fn compile_type_alias(&self, _type_alias: &TypeAlias) -> Result<()> {
        // Type aliases are typically handled at the type-checking phase
        // No LLVM code generation is needed
        Ok(())
    }

    fn compile_binary_op(&self, binary: &Binary) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        let left = self.compile_expr(&binary.left)?;
        let right = self.compile_expr(&binary.right)?;

        match (binary.op, left, right) {
            (BinaryOp::Add, 
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                Ok(self.builder.build_int_add(l, r, "addtmp")
                    .map_err(|_| CodegenError::UnsupportedExpression)?.into())
            },
            (BinaryOp::Sub,
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                Ok(self.builder.build_int_sub(l, r, "subtmp")
                    .map_err(|_| CodegenError::UnsupportedExpression)?.into())
            },
            (BinaryOp::Mul,
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                Ok(self.builder.build_int_mul(l, r, "multmp")
                    .map_err(|_| CodegenError::UnsupportedExpression)?.into())
            },
            (BinaryOp::Gt,
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                let cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::SGT,
                    l,
                    r,
                    "gttmp"
                ).map_err(|_| CodegenError::UnsupportedExpression)?;
                Ok(cmp.into())
            },
            (BinaryOp::Lt,
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                let cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLT,
                    l,
                    r,
                    "lttmp"
                ).map_err(|_| CodegenError::UnsupportedExpression)?;
                Ok(cmp.into())
            },
            (BinaryOp::Le,
             inkwell::values::BasicValueEnum::IntValue(l),
             inkwell::values::BasicValueEnum::IntValue(r)) => {
                let cmp = self.builder.build_int_compare(
                    inkwell::IntPredicate::SLE,
                    l,
                    r,
                    "letmp"
                ).map_err(|_| CodegenError::UnsupportedExpression)?;
                Ok(cmp.into())
            },
            _ => Err(CodegenError::UnsupportedExpression.into()),
        }
    }

    fn compile_unary_op(&self, unary: &Unary) -> Result<inkwell::values::BasicValueEnum<'ctx>> {
        let expr = self.compile_expr(&unary.expr)?;

        match (unary.op, expr) {
            (UnaryOp::Neg, 
             inkwell::values::BasicValueEnum::IntValue(v)) => {
                Ok(self.builder.build_int_neg(v, "negtmp")
                    .map_err(|_| CodegenError::UnsupportedExpression)?.into())
            },
            // Add other unary operations as needed
            _ => Err(CodegenError::UnsupportedExpression.into()),
        }
    }

    fn ensure_insertion_block(&self) -> Result<()> {
        if self.builder.get_insert_block().is_none() {
            // Create a dummy function if we don't have one
            let void_type = self.context.void_type();
            let fn_type = void_type.fn_type(&[], false);
            let function = self.module.add_function("__temp", fn_type, None);
            let basic_block = self.context.append_basic_block(function, "entry");
            self.builder.position_at_end(basic_block);
        }
        Ok(())
    }

    pub fn get_function<T: UnsafeFunctionPointer>(&self, name: &str) -> Result<JitFunction<T>> {
        unsafe {
            self.execution_engine
                .get_function::<T>(name)
                .map_err(|_| CodegenError::FunctionNotFound.into())
        }
    }

    pub fn dump_module(&self) -> String {
        self.module.print_to_string().to_string()
    }

    pub fn get_execution_engine(&self) -> &ExecutionEngine<'ctx> {
        &self.execution_engine
    }
}

impl<'ctx> Drop for CodeGen<'ctx> {
    fn drop(&mut self) {
        // Clear all references before dropping
        self.variables.borrow_mut().clear();
        self.struct_types.borrow_mut().clear();
        self.loop_stack.borrow_mut().clear();
    }
}


