// Simple 8-bit adder - Hello World of Verilog
// This is combinational logic: output changes instantly when inputs change

module adder (
    input  [7:0] a,      // 8-bit input a
    input  [7:0] b,      // 8-bit input b
    output [8:0] sum     // 9-bit output (extra bit for overflow)
);

    // This is continuous assignment - purely combinational
    // No clock, no state - just wires
    assign sum = a + b;

endmodule
