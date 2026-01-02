# TinyGrad for Verilog - Architecture

## Goal
Build a neural network inference engine in pure Verilog, then add training.

## Tools
- **Icarus Verilog** (`iverilog`) - open source Verilog simulator
- **GTKWave** - waveform viewer for debugging

## Phase 1: Inference

### What We're Building
A hardware circuit that computes neural network forward passes. No CPU, no software - just gates and wires.

### Core Concepts

**Fixed-Point Arithmetic**
- FPGAs don't have native float - we use fixed-point (e.g., Q8.8 = 8 integer bits, 8 fractional bits)
- Multiply two Q8.8 numbers → Q16.16, then truncate back to Q8.8

**Hardware vs Software Mindset**
- Software: sequential instructions, one at a time
- Hardware: everything happens in parallel, all at once
- A neuron `y = relu(w*x + b)` is literally wires connecting a multiplier → adder → comparator

### Module Hierarchy (Inference)

```
top.v                    # Top-level test harness
├── network.v            # The neural network
│   ├── neuron.v         # Single neuron: y = activation(sum(w*x) + b)
│   │   ├── mac.v        # Multiply-accumulate unit
│   │   └── relu.v       # ReLU activation: max(0, x)
│   └── layer.v          # Collection of neurons
└── testbench.v          # Simulation driver
```

### Data Flow (Single Neuron)
```
inputs x[0..n] ──┐
                 │
weights w[0..n] ─┼──▶ [MAC] ──▶ [+bias] ──▶ [ReLU] ──▶ output
                 │
                 ▼
            (multiply-accumulate:
             sum = w[0]*x[0] + w[1]*x[1] + ... + w[n]*x[n])
```

## Phase 2: Training (Later)

Training requires:
1. Forward pass (inference)
2. Loss computation
3. Backward pass (gradient calculation)
4. Weight updates

This is harder because:
- Need to store intermediate activations
- Backward pass requires different dataflow
- Weight updates need read-modify-write to memory

## Development Plan (~10 hours)

| Hour | Task |
|------|------|
| 1 | Install Icarus Verilog, write hello world (basic adder) |
| 2 | Implement fixed-point multiply module |
| 3 | Implement MAC (multiply-accumulate) unit |
| 4 | Implement ReLU activation |
| 5 | Build single neuron from MAC + ReLU |
| 6 | Build a layer (multiple neurons) |
| 7 | Build 2-layer network, hardcoded weights |
| 8 | Testbench: feed inputs, verify outputs |
| 9 | (If time) Python codegen for weights |
| 10 | (If time) Start on backward pass |

## MAC Module Timing Diagram

This shows how the testbench, clock, and MAC module interact:

```
TIME ────────────────────────────────────────────────────────────────────────▶

CLOCK           ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐   ┌───┐
            ────┘   └───┘   └───┘   └───┘   └───┘   └───┘   └───
                 ^       ^       ^       ^       ^
                 │       │       │       │       │
                edge 1  edge 2  edge 3  edge 4  edge 5


TESTBENCH   ════════════════════════════════════════════════════════════════
SETS:       │ x=1,w=0.5 │ x=2,w=0.5 │ x=3,w=0.5 │ x=4,w=0.5 │  (done)
            │ start=1   │ start=0   │ start=0   │ start=0   │
            ════════════════════════════════════════════════════════════════


FIXED_MUL   ════════════════════════════════════════════════════════════════
(combinational, instant)
product:    │    0.5    │    1.0    │    1.5    │    2.0    │    2.0
            ════════════════════════════════════════════════════════════════
                 ▲           ▲           ▲           ▲
                 │           │           │           │
            always ready, updates immediately when x/w change


MAC STATE   ════════════════════════════════════════════════════════════════
(captured at clock edges)

acc:            0 ──▶ 0.5 ──▶ 1.5 ──▶ 3.0 ──▶ 5.0
                      ^       ^       ^       ^
                      │       │       │       │
                   edge 1  edge 2  edge 3  edge 4
                   (start)  (+1.0)  (+1.5)  (+2.0)

count:          0 ──▶ 1 ────▶ 2 ────▶ 3 ────▶ 4

done:           0 ──▶ 0 ────▶ 0 ────▶ 0 ────▶ 1
            ════════════════════════════════════════════════════════════════
```

### Key insight: Combinational vs Sequential

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              MAC MODULE                                  │
│                                                                         │
│  COMBINATIONAL (no clock, always computing)                             │
│  ┌─────────────────────────────────────────┐                            │
│  │                                         │                            │
│  │   x ────┐                               │                            │
│  │         ├───▶ [fixed_mul] ───▶ product ─┼──┐                         │
│  │   w ────┘                               │  │                         │
│  │                                         │  │                         │
│  └─────────────────────────────────────────┘  │                         │
│                                               │                         │
│  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─│─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─   │
│                                               │                         │
│  SEQUENTIAL (updates only on clock edge)      ▼                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                                                                 │   │
│  │   product ──▶ [+] ──▶ [acc flip-flop] ──▶ acc (output)          │   │
│  │               ▲              │                                  │   │
│  │               └──────────────┘                                  │   │
│  │                (feedback)                                       │   │
│  │                                                                 │   │
│  │   start ─────▶ [control logic] ──▶ count, done                  │   │
│  │   clk ───────▶                                                  │   │
│  │                                                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### The dance:

1. **Testbench** sets x, w, start (just changing wire values)
2. **fixed_mul** instantly computes product (combinational)
3. **Clock edge** hits
4. **MAC flip-flops** capture: acc gets acc+product, count increments
5. **Testbench** changes x, w for next iteration
6. **fixed_mul** instantly updates product
7. Repeat until done

## TODOs

- [ ] Add saturation logic to `fixed_mul.v` - currently overflows silently for large values (e.g., 16.0 * 16.0). Should clamp to max/min instead of wrapping.

## File Structure

```
/
├── ARCHITECTURE.md      # This file
├── src/
│   ├── mac.v           # Multiply-accumulate
│   ├── relu.v          # ReLU activation
│   ├── neuron.v        # Single neuron
│   ├── layer.v         # Layer of neurons
│   └── network.v       # Full network
├── test/
│   └── tb_*.v          # Testbenches
└── Makefile            # Build/simulate commands
```
