module tb_layer_debug;
  reg clk;
  reg start;

  reg signed [15:0] x_vals;
  reg signed [15:0] w_vals;
  reg signed [15:0] b_vals;

  wire signed [15:0] y;
  wire done;

  layer #(
    .N(1),
    .M(1)
   ) uut (
    .clk(clk),
    .start(start),
    .x(x_vals),
    .w(w_vals),
    .b(b_vals),
    .y(y),
    .done(done)
  );

  always #5 clk = ~clk;

  initial begin
    $display("=== Layer Debug Test ===");

    x_vals = 16'd512;  // 2.0 in Q8.8
    w_vals = 16'd256;  // 1.0 in Q8.8
    b_vals = 16'd0;

    clk = 0;
    start = 0;

    $monitor("t=%0t clk=%b start_d=%b neuron_start=%b neuron_x=%d mac_start=%b mac_count=%d mac_acc=%d", 
             $time, clk, uut.start_delayed, 
             uut.genblk1[0].neuron_inst.start,
             uut.genblk1[0].neuron_inst.x,
             uut.genblk1[0].neuron_inst.mac_inst.start,
             uut.genblk1[0].neuron_inst.mac_inst.count,
             uut.genblk1[0].neuron_inst.mac_inst.acc);

    #20;
    start = 1;
    #10;
    start = 0;

    #100;
    $finish;
  end
endmodule
