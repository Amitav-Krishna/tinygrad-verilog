#!/usr/bin/env python3
"""Demo: tinygrad Verilog backend"""
import os
os.environ["VERILOG"] = "1"

from tinygrad import Tensor

print("=== tinygrad Verilog Backend Demo ===\n")

# Simple addition
a = Tensor([1.0, 2.0, 3.0, 4.0])
b = Tensor([5.0, 6.0, 7.0, 8.0])
print(f"a = {a.numpy()}")
print(f"b = {b.numpy()}")
print(f"a + b = {(a + b).numpy()}")
print(f"a * b = {(a * b).numpy()}")
print(f"a - b = {(a - b).numpy()}")

print("\n=== Generated Verilog ===")
print(open("/tmp/tinygrad_verilog/kernel.v").read())
