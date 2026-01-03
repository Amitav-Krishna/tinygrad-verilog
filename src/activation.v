module activation (
  input wire signed [15:0] weight_product,
  input wire signed [15:0] b, // Bias
  input wire start,
  output wire signed [15:0] activation,
  output wire done
);
  wire signed [15:0] preactivation;

  assign preactivation = weight_product + b;
  relu relu_inst (
    .y(preactivation),
    .activation(activation)
    );
  assign done = start;
endmodule

    


