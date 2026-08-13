1. **Does `compile` call `commit` / `waitUntilCompleted`?**
No. `compile_metal()` calls `tape->eval_metal(*this, EvalType::COMPILE_TRACE)`. This traverses the tape and encodes commands into `GlobalGPUManager.gCommandEncoder`, but it **never** calls `commit` or `waitUntilCompleted`.

2. **Does `execute_trace` create a fresh `MTLCommandBuffer` or reuse the one from `compile`?**
It reuses the exact same command buffer from `compile_metal`. In `GPUManager.h`, `getCommandBuffer()` only creates a new buffer if `gCommandBuffer` is `nil`. Because `execute_metal()` and `eval_metal()` do not reset `gCommandBuffer = nil` after committing, the global state holds onto the old buffer. 

3. **Is the command buffer in a valid re-submittable state when `execute_trace` runs?**
No. In Metal, an `MTLCommandBuffer` is strictly **single-use**. Once you call `commit` on it (which happens in the first call to `execute_metal()`), it cannot be committed again. On subsequent calls to the JIT lambda, `execute_metal` attempts to re-commit the already-completed buffer. This is invalid and fails, which is why the GPU never actually runs the new inputs, leaving the output buffer populated with the results of the first successful run (the sample data).

**The Solution:**
To make JIT execution work without `MTLIndirectCommandBuffer` (which is highly complex), you must re-encode the commands into a fresh `MTLCommandBuffer` on every execution. 
- `EvalType::COMPILE_TRACE` should be used strictly to initialize pipeline states and allocate buffers (no command encoding).
- `EvalType::EXEC_TRACE` should encode the commands and commit.
- `execute_metal` and `eval_metal` must reset `GlobalGPUManager.gCommandBuffer` and `gCommandEncoder` to `nil` after `commit` to ensure a fresh buffer for the next pass.
