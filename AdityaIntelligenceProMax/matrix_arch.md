# Matrix Library Architecture

## Core Design Philosophy

Build a lazy-evaluated, GPU-accelerated matrix/tensor library from scratch. Graph is built once, compiled once, executed many times with minimal overhead. Inspired by MLX/JAX but with a persistent compiled graph model closer to TensorFlow 1.x sessions.

---

## Class Structure

### `matrix`

The **edge** of the computational graph. Every matrix is a node-handle that owns or shares a data buffer.

```
class matrix {
    array_descriptor array_desc;

    void*                    buffer       // CPU data buffer (shared via refcount)
    size_t                   total_size
    id<MTLBuffer>            metalBuffer  // Metal GPU buffer
    std::atomic<uint32_t>*   refCount     // shared refcount for buffer ownership
    Primitive*               tape         // pointer to primitive that produced this matrix (null = leaf)

    dtype                    type
    uint32_t                 dims
    uint8_t                  flags        // NON_OWNERSHIP_FLAG, NON_CONTIGUOUS_FLAG, etc.
}
```

Key properties:
- Matrices share underlying buffers via `refCount` — copying a matrix copies the handle, not the data
- `tape == nullptr` means leaf node (input data, no producer primitive)
- `flags` tracks ownership and contiguity for safe buffer management
- `dims` stores shape; broadcasting uses `std::lcm` of operand dims

---

### `Primitive`

The **node** of the computational graph. Stores its two input matrices by value (but they share buffers with the rest of the graph via refcount).

```
class Primitive {
    matrix a               // left input (shared buffer copy)
    matrix b               // right input (shared buffer copy)
    BroadcastDescriptor* desc_a
    BroadcastDescriptor* desc_b
    uint8_t* out_buffer    // primitive-level cache of output buffer ptr

    virtual eval_cpu(matrix& out)
    virtual eval_metal(matrix& out)
    virtual build_trace_cpu(matrix& out)
    virtual build_trace_metal(matrix& out)
    virtual execute_trace_cpu(matrix& out)
    virtual execute_trace_metal(matrix& out)
    virtual vjp(grad_out) -> vector<matrix>   // reverse-mode autodiff
    virtual jvp(tangents) -> matrix           // forward-mode autodiff
}
```

Concrete subclasses: `AdditionPrimitive`, `SubtractionPrimitive`, `MultiplicationPrimitive`, `DivisionPrimitive`, etc.

The graph structure is therefore:

```
matrix → Primitive → matrix (a)
                   → matrix (b)
```

A diamond dependency looks like:

```
leaf_a ──→ AddPrim ──→ result_b ──→ AddPrim ──→ result_d
       ╲                                        ↑
        ╲──→ AddPrim ──→ result_c ──────────────┘
```

Both `result_b`'s and `result_c`'s primitives hold value-copies of `leaf_a`, but all copies share the same underlying buffer and the same `Primitive* tape` pointer on `leaf_a`.

---

### `BroadcastDescriptor`

Stores the broadcasting metadata for an operand so the Metal kernel knows how to index into it when shapes don't match.

---

## Graph Lifecycle

### 1. Build Phase (graph construction)

Operator overloads build the graph by creating primitives:

```
operator+ → add(a, b) → update_trace(a), update_trace(b) → AdditionPrimitive(a, b) → result matrix
```

- `operator+` takes `const matrix&` and `const_cast`s to non-const for `add()`
- `add()` calls `update_trace` on both inputs (checks if buffer is up to date with the graph)
- `AdditionPrimitive` copies both inputs by value (shared buffer semantics)
- Result matrix is returned with `tape` pointing to the new primitive

### 2. One-off Execution (`eval_cpu` / `eval_metal`)

For single-use evaluations — allocates and executes each node on the fly without building a persistent tape:

```
eval_metal(out)
    → check primitive.traced, skip if already done
    → recurse into tape->a, tape->b
    → allocate buffers for this node
    → dispatch Metal kernel
    → set primitive.traced = true
    → reset all traced = false after full traversal
```

- No DFS visited set — uses `Primitive::traced` bool instead
- Traverses the linked list structure directly (matrix → tape → a, b)
- Diamond deps handled by `traced` flag — second encounter is a no-op
- Requires a reset pass after each eval to clear all `traced` flags

### 3. Compile Phase (`build_trace`)

Traverses the graph, allocates all intermediate buffers, binds Metal buffers, builds command encoder:

```
build_trace
    → traverse graph via traced bool
    → allocate buffers for all intermediate nodes
    → bind Metal buffers
    → build command encoder once
    → reset traced flags
```

- Same `traced` bool mechanism as eval for deduplication
- After this call, all buffers allocated, command encoder ready, no further allocation needed

### 4. Execute Phase (`execute_trace`)

Re-runs the compiled graph on existing buffers with zero allocation:

```
execute_trace
    → traverse graph via traced bool
    → dispatch kernels on pre-bound buffers
    → reset traced flags
```

- To rerun on same or different data: swap input buffers, call `execute_trace` again
- Still traverses the linked list every run (traced bool, not flat tape)

---

## Graph Traversal

### Current Implementation

Uses a `bool traced` on each `Primitive` for deduplication across all three paths:

```cpp
// on Primitive base class
bool traced = false;
```

```cpp
void matrix::execute_trace() {
    if (!tape) return;                  // leaf node
    if (tape->traced) return;           // already visited (diamond case)
    tape->traced = true;

    tape->a.execute_trace();            // recurse into inputs
    tape->b.execute_trace();

    tape->execute_trace_metal(*this);   // execute after inputs

    // caller resets all traced = false after full traversal
}
```

Limitation: requires a reset pass after every traversal to clear `traced` flags — O(nodes) extra work.

### Target Implementation (backlog)

Replace `traced` bool with:

- **`eval` / `build_trace`**: `unordered_set<Primitive*> visited` passed through traversal — lives on stack, dies when done, zero cleanup needed
- **`execute_trace`**: flat `vector<matrix*> _tape` built once by `build_trace`, execute is a tight linear loop with no traversal at all

```cpp
// target execute_trace
void matrix::execute_trace() {
    for (auto* m : _tape)               // flat linear loop, cache friendly
        m->tape->execute_trace_metal(*m);
}
```

Diamond deduplication in target: visited set keys on `Primitive*` — since all matrix copies sharing the same node have the same `tape` pointer, second encounter is a no-op. No reset pass needed.

---

## Memory Model

- **Buffer sharing**: copying a `matrix` copies the handle; `refCount` tracks how many handles share the buffer
- **`NON_OWNERSHIP_FLAG`**: matrix does not own the buffer (e.g. slices, views)
- **`NON_CONTIGUOUS_FLAG`**: buffer is not contiguous in memory (e.g. after transpose/slice)
- **Metal buffers**: separate `id<MTLBuffer>` per matrix, managed alongside CPU buffer
- **Primitive output buffer**: `out_buffer` on primitive caches the output pointer for use in Metal kernels
- No allocation during `execute_trace` — all buffers fixed after `build_trace`

### Buffer Sharing via Primitive (vs MLX approach)

The problem: matrices are stored by value inside primitives and initially have `buffer = nullptr`. So in:

```cpp
matrix c = a + b;
matrix d = c + c;
```

`d`'s `AdditionPrimitive` holds two value-copies of `c`, both with `buffer = nullptr`. Without any sharing mechanism these would be treated as distinct nodes, allocated separately, and evaluated twice — giving them different buffers and incorrect results.

**MLX's solution**: wrap the buffer pointer in a separate ref-counted data struct so even null-buffer arrays share the same data struct pointer. Works, but adds one extra indirection on every buffer access.

**This library's solution**: store `out_buffer`, `out_metal_buffer`, and `out_refcount` directly on the `Primitive` base class. Since both copies of `c` point to the same `MulPrimitive*` via `matrix::tape`, the primitive itself becomes the shared state anchor:

```
d = c + c

AdditionPrimitive (d)
    ├── matrix c copy 1  →  tape →  MulPrimitive { out_buffer, out_metal_buffer, out_refcount }
    └── matrix c copy 2  →  tape →  MulPrimitive { same primitive, same fields }
```

During `build_trace`:
1. copy 1 hits `MulPrimitive`, `out_buffer == nullptr` → allocate, store in primitive fields, write back to copy 1
2. copy 2 hits same `MulPrimitive`, `traced == true` → skip entirely, or if reached another way sees `out_buffer != nullptr` → bind existing buffer, no reallocation

**Tradeoff vs MLX**: the extra indirection only occurs during `build_trace` when checking `primitive->out_buffer`. During `execute_trace` the matrix already has its buffer pointer directly — zero extra indirection on the hot path.

---

## Operator Pipeline

```
operator+(const matrix& a, const matrix& b)
    → const_cast<matrix&> both
    → type check (throw if mismatch)
    → result = matrix(lcm(a.dims, b.dims), a.type)   // output shell
    → a_.add(b_, result)                              // builds AdditionPrimitive
    → return result
```

Same pattern for `-`, `*`, `/`.

---

## Autograd

Each primitive implements:
- `vjp(grad_out)` — reverse mode (backprop), returns vector of input gradients
- `jvp(tangents)` — forward mode, returns output tangent

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Matrix = edge, Primitive = node | Primitives own their inputs; output is the matrix holding the primitive |
| Inputs stored by value in Primitive | Buffer sharing via refcount makes this cheap; no dangling pointer risk |
| `traced` bool on Primitive (current) | Simple dedup across all paths; requires reset pass after each traversal |
| Flat tape + visited set (target) | `execute_trace` becomes a tight loop, no reset pass, no traversal overhead |
| `const_cast` at operator level | Single overload handles lvalues and rvalues; no `&&` combinatorial explosion |
| `update_trace` in `add()` | Buffer freshness check happens where the primitive is built, not at operator level |
| No dirty region tracking | All major frameworks (MLX, JAX, PyTorch) do full reruns; kernel dispatch cost dominates traversal cost |

---

## Comparison to MLX

| | MLX | This library (current) | This library (target) |
|---|---|---|---|
| Graph lifetime | Throwaway (rebuilt each eval) | Persistent | Persistent |
| Execution model | Fresh DFS every `mx::eval()` | Recursive traversal + `traced` bool | Flat tape, linear loop |
| Deduplication | visited set (stack local) | `traced` bool + reset pass | visited set / flat tape |
| Dirty regions | None | None (full rerun) | None (full rerun) |
| Input reuse | N/A | Swap buffers, call `execute_trace` | Swap buffers, call `execute_trace` |
| Autodiff | vjp/jvp | vjp/jvp | vjp/jvp |
| Backend | Metal/CPU | Metal/CPU | Metal/CPU |
