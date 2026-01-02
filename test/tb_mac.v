// Testbench for MAC unit

`timescale 1ns/1ps

module tb_mac;

    // Clock and control
    reg clk;
    reg rst;
    reg start;
    
    // Data inputs
    reg signed [15:0] x;
    reg signed [15:0] w;
    
    // Outputs
    wire signed [15:0] acc;
    wire done;

    // Instantiate MAC with N=4 inputs
    mac #(.N(4)) uut (
        .clk(clk),
        .rst(rst),
        .start(start),
        .x(x),
        .w(w),
        .acc(acc),
        .done(done)
    );

    // Clock generation: 10ns period
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

    // Test vectors: x = [1, 2, 3, 4], w = [0.5, 0.5, 0.5, 0.5]
    // Expected: 1*0.5 + 2*0.5 + 3*0.5 + 4*0.5 = 5.0
    reg signed [15:0] x_vals [0:3];
    reg signed [15:0] w_vals [0:3];
    integer i;

    initial begin
        $display("=== MAC Unit Test ===");
        
        // Initialize test vectors
        x_vals[0] = real_to_q8(1.0);
        x_vals[1] = real_to_q8(2.0);
        x_vals[2] = real_to_q8(3.0);
        x_vals[3] = real_to_q8(4.0);
        
        w_vals[0] = real_to_q8(0.5);
        w_vals[1] = real_to_q8(0.5);
        w_vals[2] = real_to_q8(0.5);
        w_vals[3] = real_to_q8(0.5);

        // Initialize
        clk = 0;
        rst = 1;
        start = 0;
        x = 0;
        w = 0;

        // Release reset
        #20;
        rst = 0;
        
        // Start computation
        #10;
        start = 1;
        x = x_vals[0];
        w = w_vals[0];
        #10;
        start = 0;

        // Feed remaining values
        for (i = 1; i < 4; i = i + 1) begin
            x = x_vals[i];
            w = w_vals[i];
            #10;
        end

        // Wait for done
        #20;
        
        $display("  x = [1.0, 2.0, 3.0, 4.0]");
        $display("  w = [0.5, 0.5, 0.5, 0.5]");
        $display("  Result: %.3f (expected 5.0)", q8_to_real(acc));
        $display("  Done: %b", done);

        // Second test: x = [1, -1, 2, -2], w = [1, 1, 1, 1]
        // Expected: 1 - 1 + 2 - 2 = 0
        $display("");
        $display("--- Second test ---");
        
        x_vals[0] = real_to_q8(1.0);
        x_vals[1] = real_to_q8(-1.0);
        x_vals[2] = real_to_q8(2.0);
        x_vals[3] = real_to_q8(-2.0);
        
        w_vals[0] = real_to_q8(1.0);
        w_vals[1] = real_to_q8(1.0);
        w_vals[2] = real_to_q8(1.0);
        w_vals[3] = real_to_q8(1.0);

        // Start new computation
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
        
        $display("  x = [1.0, -1.0, 2.0, -2.0]");
        $display("  w = [1.0, 1.0, 1.0, 1.0]");
        $display("  Result: %.3f (expected 0.0)", q8_to_real(acc));

        $display("");
        $display("=== Done ===");
        $finish;
    end

endmodule
