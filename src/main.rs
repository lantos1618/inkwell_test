use inkwell::{context::Context, OptimizationLevel};
use anyhow::Result;
use inkwell_test::llvm_codegen::CodeGen;

fn main() -> Result<()> {
    let context = Context::create();
    let module = context.create_module("sum");
    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None)
        .map_err(|e| anyhow::anyhow!("Failed to create execution engine: {}", e))?;
    let builder = context.create_builder();
    
    let codegen = CodeGen::new(&context, module, builder, execution_engine);

    let sum = codegen.jit_compile_sum()
        .ok_or_else(|| anyhow::anyhow!("Unable to JIT compile `sum`"))?;

    let x = 1u64;
    let y = 2u64;
    let z = 3u64;

    unsafe {
        println!("{} + {} + {} = {}", x, y, z, sum.call(x, y, z));
        assert_eq!(sum.call(x, y, z), x + y + z);
    }

    Ok(())
}