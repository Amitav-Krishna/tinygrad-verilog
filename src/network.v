module layer #(
  parameter N = 4,
  parameter M = 4,
) (
  input wire clk,
  input wire start,
  input wire signed [(N*16)-1:0] x,
  input wire signed [((M*N)*16)-1:0] w,
  input wire signed [(M*16)-1:0] b,
  output wire signed [(M*16)-1:0] y,
  output wire done
);
  wire signed [(M*16)-1:0] activations;


