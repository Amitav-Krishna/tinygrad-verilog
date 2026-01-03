
`timescale 1ns/1ps;

module tb_neuron;
  reg clk;
  reg start;

  reg signed [15:0] x;
  reg signed [15:0] w;
  reg signed [15:0] b;

  wire signed [15:0] activation;
  wire done;

  neuron #(.N(4)) uut (
    .clk(clk),
    .start(start),
    .x(x),
    .w(w),
    .b(b),
    .activation(activation),
    .done(done)
  );

  always #5 clk = ~clk;

  // Helper functions
  function real q8_to_real;
      input signed [15:0] q;
      begin
          q8_to_real = $itor(q) / 256.0;
      end
  endfunction

  function signed [15:0] real_to_q8;
      input real r;
      begin
          real_to_q8 = $rtoi(r * 256.0);
      end
  endfunction
  
  reg signed [15:0] x_vals [0:3];
  reg signed [15:0] w_vals [0:3];
  integer i;

  initial begin
    $display("=== Neuron Unit Test ===");


      x_vals[0] = real_to_q8(1.0);
      x_vals[1] = real_to_q8(2.0);
      x_vals[2] = real_to_q8(3.0);
      x_vals[3] = real_to_q8(4.0);
      
      w_vals[0] = real_to_q8(0.5);
      w_vals[1] = real_to_q8(1.5);
      w_vals[2] = real_to_q8(2.5);
      w_vals[3] = real_to_q8(3.5);

      b = real_to_q8(5.0);

      clk = 00;

      start = 0;
      x = 0;
      w = 0;

      #20;

      #10;
      start = 1;
      x = x_vals[0];
      w = w_vals[0];

      #10;
      start = 0;

      for (i = 1; i < 4; i = i + 1) begin
          x = x_vals[i];
          w = w_vals[i];
          #10;
      end

      #20;
      
      $display("  x = [1.0, 2.0, 3.0, 4.0]");
      $display("  w = [0.5, 1.5, 2.5, 3.5]");
      $display("  Result: %.3f (expected 30.0)", q8_to_real(activation));
      $display("  Done: %b", done);
    end
    endmodule





