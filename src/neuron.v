module neuron #(
  parameter N = 4 
) (
  input wire clk,
  input wire start,
  input wire signed [15:0] x,
  input wire signed [15:0] w,
  input wire signed [15:0] b,
  output wire signed [15:0] activation,
  output wire done
  );

  wire signed [15:0] acc;
  wire rst;
  wire done_mac;
  assign rst = 0;


  mac #(.N(N)) mac_inst (
    .clk(clk),
    .rst(rst),
    .start(start),
    .x(x),
    .w(w),
    .acc(acc),
    .done(done_mac)
    );
   
  activation activation_inst (
    .start(done_mac),
    .weight_product(acc),
    .b(b),
    .activation(activation),
    .done(done)
    );
endmodule


  

