// Test heterogeneous network: 2 -> 3 -> 1
module tb_network_hetero;

    reg clk;
    reg start;

    // 2 inputs
    reg signed [31:0] x;
    // 9 weights: 3*2 + 1*3 = 6 + 3
    reg signed [143:0] w;
    // 4 biases: 3 + 1
    reg signed [63:0] b;
    // 1 output
    wire signed [15:0] y;
    // 6 activations: 2 + 3 + 1
    wire signed [95:0] acts;
    wire done;

    network #(
        .NUM_LAYERS(2),
        .LAYER_SIZES({16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd1, 16'd3, 16'd2})
    ) uut (
        .clk(clk),
        .start(start),
        .x(x),
        .w(w),
        .b(b),
        .y(y),
        .intermediate_states(acts),
        .done(done)
    );

    always #5 clk = ~clk;

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

    initial begin
        $display("=== Heterogeneous Network Test (2->3->1) ===");

        clk = 0;
        start = 0;

        // Input: [1.0, 0.5]
        x = {real_to_q8(0.5), real_to_q8(1.0)};

        // Layer 0 weights (3 neurons, 2 inputs each = 6 weights)
        // Stored column-major for MAC: [col0: n0,n1,n2][col1: n0,n1,n2]
        w[0*16 +: 16] = real_to_q8(0.1);   // n0, input0
        w[1*16 +: 16] = real_to_q8(0.2);   // n1, input0
        w[2*16 +: 16] = real_to_q8(0.3);   // n2, input0
        w[3*16 +: 16] = real_to_q8(0.4);   // n0, input1
        w[4*16 +: 16] = real_to_q8(0.5);   // n1, input1
        w[5*16 +: 16] = real_to_q8(0.6);   // n2, input1

        // Layer 1 weights (1 neuron, 3 inputs = 3 weights)
        w[6*16 +: 16] = real_to_q8(0.7);   // n0, input0
        w[7*16 +: 16] = real_to_q8(0.8);   // n0, input1
        w[8*16 +: 16] = real_to_q8(0.9);   // n0, input2

        // Layer 0 biases (3 neurons)
        b[0*16 +: 16] = real_to_q8(0.1);
        b[1*16 +: 16] = real_to_q8(0.1);
        b[2*16 +: 16] = real_to_q8(0.1);

        // Layer 1 bias (1 neuron)
        b[3*16 +: 16] = real_to_q8(0.2);

        #20;
        start = 1;
        #10;
        start = 0;

        wait(done);
        #10;

        $display("Input: [%.3f, %.3f]", q8_to_real(x[0 +: 16]), q8_to_real(x[16 +: 16]));
        $display("Layer 0 output: [%.3f, %.3f, %.3f]",
            q8_to_real(acts[32 +: 16]),
            q8_to_real(acts[48 +: 16]),
            q8_to_real(acts[64 +: 16]));
        $display("Final output: %.3f", q8_to_real(y));
        $display("Done: %b", done);

        $display("=== Test Complete ===");
        $finish;
    end

endmodule
