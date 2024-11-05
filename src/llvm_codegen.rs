use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::OptimizationLevel;


use anyhow::Result;
use crate::ast::*;
use crate::error::CodegenError;

/// Convenience type alias for the `sum` function.
type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: ExecutionEngine<'ctx>,
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
            Stmt::Expr(expr) => self.compile_expr(expr),
            Stmt::Return(ret) => self.compile_return(ret),
            Stmt::Break => self.compile_break(),
            Stmt::Continue => self.compile_continue(),
            Stmt::StructDecl(struct_decl) => self.compile_struct_decl(struct_decl),
            Stmt::StructDef(struct_def) => self.compile_struct_def(struct_def),
            Stmt::EnumDecl(enum_decl) => self.compile_enum_decl(enum_decl),
            Stmt::EnumDef(enum_def) => self.compile_enum_def(enum_def),
            Stmt::TypeAlias(type_alias) => self.compile_type_alias(type_alias),
            Stmt::FuncCall(func_call) => self.compile_func_call(func_call),
        }
    }

    /// Translates AST types to LLVM types.
    fn into_llvm_type(&self, ast_type: &AstType ) -> Result<inkwell::types::BasicTypeEnum<'ctx>> {
        match ast_type {
            AstType::I8 => Ok(self.context.i8_type().into()),
            AstType::I16 => Ok(self.context.i16_type().into()),
            AstType::I32 => Ok(self.context.i32_type().into()),
            AstType::I64 => Ok(self.context.i64_type().into()),
            AstType::U8 => Ok(self.context.i8_type().into()),
            AstType::U16 => Ok(self.context.i16_type().into()),
            AstType::U32 => Ok(self.context.i32_type().into()),
            AstType::U64 => Ok(self.context.i64_type().into()),
            AstType::F32 => Ok(self.context.f32_type().into()),
            AstType::F64 => Ok(self.context.f64_type().into()),
            AstType::Bool => Ok(self.context.bool_type().into()),
            AstType::Char => Ok(self.context.i8_type().into()), // Assuming `char` is an 8-bit integer
            AstType::Void => todo!(),
            AstType::Enum(_) => todo!(),
            AstType::TypeAlias(_) => todo!(),
            AstType::String => todo!(),
            AstType::Struct(_) => todo!(),
        }
    }

    fn compile_func_decl(&self, func_decl: &FuncDecl) -> Result<()> {
        // Use `into_llvm_type` to obtain the LLVM return type
        let return_type = match &func_decl.return_type {
            Some(ast_type) => self.into_llvm_type(&ast_type)?,
            None => todo!(),
        };

        Ok(())
    }

    fn compile_func_def(&self, _func_def: &FuncDef) -> Result<()> {
        unimplemented!()
    }

    fn compile_var_decl(&self, _var_decl: &VarDecl) -> Result<()> {
        unimplemented!()
    }

    fn compile_assign(&self, _assign: &Assign) -> Result<()> {
        unimplemented!()
    }

    fn compile_block(&self, _block: &Block) -> Result<()> {
        unimplemented!()
    }

    fn compile_loop(&self, _loop_stmt: &LoopStmt) -> Result<()> {
        unimplemented!()
    }

    fn compile_if(&self, _if_stmt: &IfStmt) -> Result<()> {
        unimplemented!()
    }

    fn compile_expr(&self, _expr: &Expr) -> Result<()> {
        unimplemented!()
    }

    fn compile_return(&self, _ret: &Return) -> Result<()> {
        unimplemented!()
    }

    fn compile_break(&self) -> Result<()> {
        unimplemented!()
    }

    fn compile_continue(&self) -> Result<()> {
        unimplemented!()
    }

    fn compile_struct_decl(&self, _struct_decl: &StructDecl) -> Result<()> {
        unimplemented!()
    }

    fn compile_struct_def(&self, _struct_def: &StructDef) -> Result<()> {
        unimplemented!()
    }

    fn compile_enum_decl(&self, _enum_decl: &EnumDecl) -> Result<()> {
        unimplemented!()
    }

    fn compile_enum_def(&self, _enum_def: &EnumDef) -> Result<()> {
        unimplemented!()
    }

    fn compile_type_alias(&self, _type_alias: &TypeAlias) -> Result<()> {
        unimplemented!()
    }

    fn compile_func_call(&self, _func_call: &FuncCall) -> Result<()> {
        unimplemented!()
    }

    /// JIT compiles a sum function and returns a callable JIT function.
    pub fn jit_compile_sum(&self) -> Option<JitFunction<SumFunc>> {
        let i64_type = self.context.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = self.module.add_function("sum", fn_type, None);
        let basic_block = self.context.append_basic_block(function, "entry");

        self.builder.position_at_end(basic_block);

        // Sum the three parameters
        let x = function.get_nth_param(0)?.into_int_value();
        let y = function.get_nth_param(1)?.into_int_value();
        let z = function.get_nth_param(2)?.into_int_value();

        let sum = self.builder.build_int_add(x, y, "sum").unwrap();
        let sum = self.builder.build_int_add(sum, z, "sum_total").unwrap();

        self.builder.build_return(Some(&sum)).unwrap();

        unsafe { self.execution_engine.get_function("sum").ok() }
    }
}
