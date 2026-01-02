`timescale 1ns/1ps

module tb_relu;
  reg  signed [15:0] y;
  wire signed [15:0] activation;

  relu uut (
    .y(y),
    .activation(activation)
  );

    // Helper: convert Q8.8 to real for display
    // In simulation only - not synthesizable
    function real q8_to_real;
        input signed [15:0] q;
        begin
            q8_to_real = $itor(q) / 256.0;
        end
    endfunction

    // Helper: convert real to Q8.8
    function signed [15:0] real_to_q8;
        input real r;
        begin
            real_to_q8 = $rtoi(r * 256.0);
        end
    endfunction

  initial begin
    $display("=== ReLU Testbench ===");

    // Test case 1
    y = real_to_q8(10.0);
    #10;
    $display("relu(%.3f) = %.3f (expected 10.0)", q8_to_real(y), q8_to_real(activation));

    // Test case 2
    y = real_to_q8(-10.0);
    #10;
    $display("relu(%.3f) = %.3f (expected 0.0)",  q8_to_real(y), q8_to_real(activation));

    // Test case 3
    y = real_to_q8(0.0);
    #10;
    $display("relu(%.3f) = %.3f (expected 0.0)",  q8_to_real(y), q8_to_real(activation));
  end


endmodule
