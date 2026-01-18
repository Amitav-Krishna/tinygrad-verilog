# Verilog backend for tinygrad
# Generates Verilog code from UOps and runs through iverilog simulation

import subprocess, tempfile, struct
from pathlib import Path
from tinygrad.dtype import DType, dtypes
from tinygrad.device import Compiled, Compiler, Allocator, CompilerSet, CompilerPair
from tinygrad.uop.ops import Ops, UOp
from tinygrad.renderer import Renderer

# Directory for generated Verilog files
VERILOG_DIR = Path(tempfile.gettempdir()) / "tinygrad_verilog"
VERILOG_DIR.mkdir(exist_ok=True)

class VerilogRenderer(Renderer):
  device = "VERILOG"
  supports_float4 = False  # disable vectorization for now

  def render(self, uops: list[UOp]) -> str:
    """Convert UOps to Verilog code."""
    # O(n) - build UOp->index mapping once for O(1) lookups
    uop_to_idx = {id(u): i for i, u in enumerate(uops)}

    # Track buffers: arg -> (name, size)
    buffers: dict[int, tuple[str, int]] = {}
    # Track variable names for each UOp index
    var_names: dict[int, str] = {}

    # Track index variables and their bounds
    index_vars: dict[str, int] = {}  # var_name -> bound

    # First pass: collect buffer definitions and find loop bounds
    for i, u in enumerate(uops):
      if u.op is Ops.DEFINE_GLOBAL:
        size = u.dtype.size if hasattr(u.dtype, 'size') else 1
        buffers[u.arg] = (f"buf{u.arg}", size)
      elif u.op is Ops.SPECIAL:
        # Index variable (lidx0, gidx0, etc.)
        var_name = u.arg
        bound_idx = uop_to_idx[id(u.src[0])]
        if uops[bound_idx].op is Ops.CONST:
          index_vars[var_name] = uops[bound_idx].arg

    # Generate Verilog
    lines = []
    lines.append("module kernel;")
    lines.append("")

    # Declare buffers - use reg[63:0] for bit storage, real for computation
    for buf_idx, (name, size) in sorted(buffers.items()):
      lines.append(f"  reg [63:0] {name}_bits [0:{size-1}];")
      lines.append(f"  real {name} [0:{size-1}];")

    lines.append("")
    # Declare all index variables plus a helper for memory ops
    lines.append("  integer _i;")  # helper for memory read/write loops
    for var_name in index_vars:
      lines.append(f"  integer {var_name};")
    lines.append("")
    lines.append("  initial begin")

    # Read input buffers (as bits) then convert to real
    for buf_idx, (name, size) in sorted(buffers.items()):
      if buf_idx > 0:  # inputs (output is buf0)
        lines.append(f"    $readmemh(\"input_{buf_idx}.hex\", {name}_bits);")
        lines.append(f"    for (_i = 0; _i < {size}; _i = _i + 1)")
        lines.append(f"      {name}[_i] = $bitstoreal({name}_bits[_i]);")

    lines.append("")
    # Generate nested loops for all index variables
    # If no index vars (vectorized case), just use a simple block
    if index_vars:
      for var_name, bound in index_vars.items():
        lines.append(f"    for ({var_name} = 0; {var_name} < {bound}; {var_name} = {var_name} + 1) begin")
    else:
      lines.append("    begin  // vectorized (no loop)")

    # Second pass: generate computation
    for i, u in enumerate(uops):
      if u.op is Ops.DEFINE_GLOBAL:
        var_names[i] = buffers[u.arg][0]

      elif u.op is Ops.CONST:
        var_names[i] = str(u.arg)

      elif u.op is Ops.SPECIAL:
        # Local index variable
        var_names[i] = u.arg  # "lidx0"

      elif u.op is Ops.INDEX:
        # Array indexing: buf[idx]
        buf_idx = uop_to_idx[id(u.src[0])]
        idx_idx = uop_to_idx[id(u.src[1])]
        buf_name = var_names.get(buf_idx, "buf0")
        idx_name = var_names.get(idx_idx, "0")
        var_names[i] = f"{buf_name}[{idx_name}]"

      elif u.op is Ops.LOAD:
        # Load just references the indexed location
        src_idx = uop_to_idx[id(u.src[0])]
        var_names[i] = var_names.get(src_idx, "0")

      elif u.op is Ops.STORE:
        dest_idx = uop_to_idx[id(u.src[0])]
        val_idx = uop_to_idx[id(u.src[1])]
        val_uop = uops[val_idx]

        if val_uop.op is Ops.VECTORIZE:
          # Store each element of the vector
          dest_base = var_names.get(dest_idx, "buf0[0]")
          if '[' in dest_base:
            base_name = dest_base.split('[')[0]
            for j, src in enumerate(val_uop.src):
              elem_idx = uop_to_idx[id(src)]
              elem_val = var_names.get(elem_idx, "0")
              lines.append(f"      {base_name}[{j}] = {elem_val};")
          else:
            lines.append(f"      {dest_base} = 0; // vectorize store fallback")
        else:
          dest = var_names.get(dest_idx, "buf0[0]")
          val = var_names.get(val_idx, "0")
          lines.append(f"      {dest} = {val};")

      elif u.op is Ops.ADD:
        a_idx = uop_to_idx[id(u.src[0])]
        b_idx = uop_to_idx[id(u.src[1])]
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"({a} + {b})"

      elif u.op is Ops.MUL:
        a_idx = uop_to_idx[id(u.src[0])]
        b_idx = uop_to_idx[id(u.src[1])]
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"({a} * {b})"

      elif u.op is Ops.SUB:
        a_idx = uop_to_idx[id(u.src[0])]
        b_idx = uop_to_idx[id(u.src[1])]
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"({a} - {b})"

      elif u.op is Ops.MAX:
        a_idx = uop_to_idx[id(u.src[0])]
        b_idx = uop_to_idx[id(u.src[1])]
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"(({a} > {b}) ? {a} : {b})"

      elif u.op is Ops.CMPLT:
        a_idx = uop_to_idx[id(u.src[0])]
        b_idx = uop_to_idx[id(u.src[1])]
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"({a} < {b})"

      elif u.op is Ops.WHERE:
        cond_idx = uop_to_idx[id(u.src[0])]
        a_idx = uop_to_idx[id(u.src[1])]
        b_idx = uop_to_idx[id(u.src[2])]
        cond = var_names.get(cond_idx, "0")
        a = var_names.get(a_idx, "0")
        b = var_names.get(b_idx, "0")
        var_names[i] = f"({cond} ? {a} : {b})"

      elif u.op is Ops.CAST:
        # Type cast - for now just pass through
        src_idx = uop_to_idx[id(u.src[0])]
        var_names[i] = var_names.get(src_idx, "0")

      elif u.op is Ops.GEP:
        # Get Element Pointer - extract element from vector
        src_idx = uop_to_idx[id(u.src[0])]
        elem_idx = u.arg[0] if u.arg else 0
        src_name = var_names.get(src_idx, "0")
        # If source is a buffer index, add the element offset
        if '[' in src_name:
          # buf[0] -> buf[0 + elem_idx]
          base = src_name.split('[')[0]
          var_names[i] = f"{base}[{elem_idx}]"
        else:
          var_names[i] = f"{src_name}_{elem_idx}"

      elif u.op is Ops.VECTORIZE:
        # Combine multiple values - store each element
        # This is handled specially in STORE
        var_names[i] = f"vec_{i}"

      elif u.op in (Ops.SINK, Ops.NOOP, Ops.BARRIER):
        pass  # No-op

    # Close all nested loops (or the single begin block)
    if index_vars:
      for _ in index_vars:
        lines.append("    end")
    else:
      lines.append("    end")
    lines.append("")

    # Convert output to bits and write
    if 0 in buffers:
      name, size = buffers[0]
      lines.append(f"    for (_i = 0; _i < {size}; _i = _i + 1)")
      lines.append(f"      {name}_bits[_i] = $realtobits({name}[_i]);")
      lines.append(f"    $writememh(\"output_0.hex\", {name}_bits);")

    lines.append("    $finish;")
    lines.append("  end")
    lines.append("endmodule")

    return "\n".join(lines)


class VerilogCompiler(Compiler):
  def compile(self, src: str) -> bytes:
    """Compile Verilog source with iverilog."""
    src_path = VERILOG_DIR / "kernel.v"
    out_path = VERILOG_DIR / "kernel.vvp"

    src_path.write_text(src)

    result = subprocess.run(
      ["iverilog", "-o", str(out_path), str(src_path)],
      capture_output=True, text=True
    )

    if result.returncode != 0:
      raise RuntimeError(f"iverilog compilation failed:\n{result.stderr}\n\nSource:\n{src}")

    return str(out_path).encode()


class VerilogProgram:
  def __init__(self, name: str, lib: bytes):
    self.vvp_path = lib.decode()

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False):
    """Run the Verilog simulation."""
    import time
    st = time.perf_counter()

    # Write input buffers to hex files
    for i, buf in enumerate(bufs):
      if i > 0:  # Skip output buffer (index 0)
        hex_path = VERILOG_DIR / f"input_{i}.hex"
        self._write_hex(hex_path, buf)

    # Run simulation
    result = subprocess.run(
      ["vvp", self.vvp_path],
      capture_output=True, text=True,
      cwd=VERILOG_DIR
    )

    if result.returncode != 0:
      raise RuntimeError(f"vvp simulation failed:\n{result.stderr}")

    # Read output buffer
    if bufs:
      out_path = VERILOG_DIR / "output_0.hex"
      if out_path.exists():
        self._read_hex(out_path, bufs[0])

    return time.perf_counter() - st

  def _write_hex(self, path: Path, buf: memoryview):
    """Write buffer to hex file (one float per line as hex)."""
    with open(path, 'w') as f:
      count = len(buf) // 4  # 4 bytes per float32
      floats = struct.unpack(f'{count}f', buf[:count*4])
      for val in floats:
        # Convert float32 to float64 then to hex (Verilog real is 64-bit)
        hex_val = struct.unpack('Q', struct.pack('d', float(val)))[0]
        f.write(f"{hex_val:016x}\n")

  def _read_hex(self, path: Path, buf: memoryview):
    """Read hex file back to buffer."""
    with open(path, 'r') as f:
      lines = f.readlines()

    floats = []
    for line in lines:
      line = line.strip()
      # Skip comments and invalid lines
      if not line or line.startswith('//') or line.startswith('x'):
        continue
      try:
        hex_val = int(line, 16)
        val = struct.unpack('d', struct.pack('Q', hex_val))[0]
        floats.append(float(val))
      except (ValueError, struct.error):
        pass

    # Write back to buffer as float32, respecting buffer size
    count = min(len(floats), len(buf) // 4)
    for i in range(count):
      struct.pack_into('f', buf, i * 4, floats[i])


class VerilogAllocator(Allocator['VerilogDevice']):
  def _alloc(self, size, options):
    return memoryview(bytearray(size))

  def _copyin(self, dest, src: memoryview):
    dest[:] = src

  def _copyout(self, dest: memoryview, src):
    dest[:] = src


class VerilogDevice(Compiled):
  def __init__(self, device: str):
    super().__init__(
      device,
      VerilogAllocator(self),
      CompilerSet([CompilerPair(VerilogRenderer, VerilogCompiler)]),
      VerilogProgram
    )
