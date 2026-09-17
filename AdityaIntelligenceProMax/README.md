# WorldOf3D / AdityaIntelligenceProMax

A from-scratch 3D graphics and tensor-compute engine for macOS/iOS, built on a custom
CPU/Metal-accelerated `matrix` library that unifies everything from GPU vertex buffers to
autodiff tensors under one lightweight, DAG-based tensor type.

## Core Components

- **`matrix` engine** (`matrix.h`, `Matrix.mm`, `MatrixH.mm`) — a lightweight, copyable,
  stack-allocated tensor struct backed by a compute-graph (DAG) of primitives, with CPU and
  Metal (GPU) execution backends, autodiff (JVP/VJP), and JIT dispatch. See
  [`.agents/MatrixAPI.md`](.agents/MatrixAPI.md) for the full API surface and
  [`.agents/AddingNewOps.md`](.agents/AddingNewOps.md) for the 3-phase pipeline (frontend /
  primitive / backend) used to add new ops.
- **Memory model** — dual reference-counting (instance ownership vs. shared-buffer views) so
  matrices behave like cheap value types while safely sharing underlying GPU/CPU buffers. See
  [`.agents/MemoryManagement.md`](.agents/MemoryManagement.md).
- **DAG versioning** — a version-counter based invalidation system (replacing an earlier,
  buggy dirty-flag approach) for correctly propagating change signals through shared/diamond
  dependency graphs. See [`.agents/versioning.md`](.agents/versioning.md).

## Rendering (`SOPs/`)

The current, modern rendering path is a `GeoNode` / `Viewer` / `Controllers` scene-graph built
entirely on top of `matrix` tensors (vertex buffers, transforms, etc. are just DAG nodes):

| File | Contains |
|---|---|
| [`SOPs/GeoNode.cpp`](SOPs/GeoNode.cpp) | `Mesh`, `Material`, `Topology`, `DepthBias`, `GeoNode`/`GeoNodeImpl` — the scene graph node |
| [`SOPs/Viewer.cpp`](SOPs/Viewer.cpp) | `Viewer` — draw loop, ray hit-testing, drag/gizmo interaction |
| [`SOPs/Controllers.cpp`](SOPs/Controllers.cpp) | `CubeController`, `LineController`, `GizmoController`, `ModelController`, `GridController`, ... |
| [`SOPs/MeshPrimitives.cpp`](SOPs/MeshPrimitives.cpp) | Primitive mesh generators: `triangle`, `quad`, `cube`, `circle`, `sphere`, `cylinder`, `cone` |
| [`SOPs/GraphViewer.cpp`](SOPs/GraphViewer.cpp) | `GraphViewer : Viewer` — 2D function-plotting scene example |

Two earlier rendering systems (`Shape<T>` and `GeometryNode<T>`, still physically present in
`MatrixH.mm` / `Mods/GeometryNode.mm` and still drawn by the legacy `Renderer` path) are
superseded and frozen — see [`.agents/Legacy_Rendering.md`](.agents/Legacy_Rendering.md). New
rendering code should be modeled on `GeoNode`/`Viewer`, not on these.

## LiDAR / Perception (`FoveaMap/`, `inputs/`)

`FoveaMap` is a ground-plane perception pipeline built on `matrix`:
depth→metric XYZ reprojection using real camera intrinsics, GPU-accelerated ground-plane
RANSAC, ground/object classification, voxel-hash object clustering, a 2D classification
overlay, RGB→depth colorization, and a rolling point-cloud frame history.

`inputs/` provides live sensor capture that feeds it: `ARSessionCapture` (ARKit point cloud +
depth via Metal, iOS-only) and `GravitySensor` (CoreMotion gravity vector, used to constrain
ground-plane RANSAC to the true "down" direction).

## Streaming (`QuantumStream*`)

`QuantumStream` / `QuantumStreamUDP` provide full-frame and UDP-chunked network transport for
streaming live `matrix` DAG tensors (e.g. depth/point-cloud frames) across the wire.
**Work in progress** — streaming does not yet work correctly.

## Other Directories

- **`Algos/`** — standalone algorithm exercises/utilities (e.g. `linked_list_sum.cpp`).
- **`Mods/`** — supporting utilities and legacy modules (e.g. `Utils.h`, `GeometryNode.mm`).
- **`ComputeShaders/`** — Metal compute kernels.
- **`mediapipe/`** — MediaPipe integration.

## Agent / Contributor Docs (`.agents/`)

Guides for working in this codebase (both for humans and coding agents):

- [`AGENTS.md`](.agents/AGENTS.md) — entry point; when to consult each doc below.
- [`MatrixAPI.md`](.agents/MatrixAPI.md) — full `matrix` API reference (check before adding new methods).
- [`AddingNewOps.md`](.agents/AddingNewOps.md) — the required pipeline for adding new primitives/ops.
- [`MemoryManagement.md`](.agents/MemoryManagement.md) — the dual refcounting system.
- [`Rendering_3D.md`](.agents/Rendering_3D.md) — the modern `GeoNode`/`Viewer` rendering architecture.
- [`Legacy_Rendering.md`](.agents/Legacy_Rendering.md) — the superseded `Shape<T>`/`GeometryNode<T>` systems (do not model new code on these).
- [`Generators.md`](.agents/Generators.md) — pattern for adding new matrix generators (`zeros`, `ones`, `gaussian`, ...).
- [`versioning.md`](.agents/versioning.md) — DAG invalidation via versioning (replacing dirty flags).
- [`Bug_Fix_L3_Matrix.md`](.agents/Bug_Fix_L3_Matrix.md) — postmortem on Metal command-buffer race conditions.
