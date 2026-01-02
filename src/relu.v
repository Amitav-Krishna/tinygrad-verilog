module relu (
  input  signed [15:0] y,
  output signed [15:0] activation
);
  assign activation = y[15] ? 16'b0 : y;
endmodule
