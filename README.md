# WorldOf3D

A monorepo of C++/Objective-C++/Metal/Swift graphics, compute, and algorithm experiments for
macOS/iOS. The active, primary project is **[`AdityaIntelligenceProMax`](AdityaIntelligenceProMax/)**.

## Main Project

### [`AdityaIntelligenceProMax/`](AdityaIntelligenceProMax/)

A from-scratch 3D graphics and tensor-compute engine built on a custom CPU/Metal-accelerated
`matrix` library — a DAG-based tensor type with autodiff, unifying GPU vertex buffers, compute
tensors, and scene-graph transforms under one lightweight abstraction. Includes:

- **`matrix` compute engine** — CPU/Metal tensor DAG with autodiff (JVP/VJP) and JIT dispatch.
- **`GeoNode`/`Viewer` renderer** (`SOPs/`) — the current scene-graph and render-loop pattern.
- **`FoveaMap/`** — a LiDAR ground-plane perception pipeline (depth→XYZ reprojection, GPU
  RANSAC, object clustering) fed by ARKit/CoreMotion sensor capture in `inputs/`.
- **`QuantumStream*`** — experimental network transport for streaming live `matrix` DAG
  tensors (work in progress).

See [`AdityaIntelligenceProMax/README.md`](AdityaIntelligenceProMax/README.md) for the full
architecture writeup, and [`AdityaIntelligenceProMax/.agents/`](AdityaIntelligenceProMax/.agents/)
for detailed contributor/agent docs (API reference, how to add new ops, memory model,
rendering architecture, etc).

## Other Projects

Earlier/standalone experiments kept for reference, each a self-contained macOS/iOS app or CLI:

| Folder | What it is |
|---|---|
| [`Aditya_Intelligence/`](Aditya_Intelligence/) | An earlier iteration of the compute/tensor engine (`Algebro`, `AlgebroHeap`) that predates `AdityaIntelligenceProMax`. |
| [`Algorithms/`](Algorithms/) | Standalone algorithm exercises and a small Metal-backed renderer prototype. |
| [`ImageProcessing/`](ImageProcessing/) | A SwiftUI app with Metal image filters (`Filters.metal`). |
| [`ParticleSystem/`](ParticleSystem/) | A Metal GPU particle system demo. |
| [`RotatingCube/`](RotatingCube/) | A minimal Metal "rotating cube" starter demo. |
| [`Scene2D/`](Scene2D/) | A minimal 2D Metal scene demo. |
| [`Video Editor/`](Video%20Editor/) / [`VideoEditor/`](VideoEditor/) | Metal/AVFoundation-based video editor prototypes. |
| [`WorldOf3D/`](WorldOf3D/) | An earlier 3D/GPU-compute prototype that predates the `AdityaIntelligenceProMax` engine. |
| [`metal-cpp/`](metal-cpp/) | Apple's `metal-cpp` bindings (vendored dependency). |
| [`test/`](test/) | Scratch/test harness. |

## Getting Started

Open [`WorldOf3D.xcodeproj`](WorldOf3D.xcodeproj) in Xcode and select the
`AdityaIntelligenceProMax` scheme to build and run the main app.
