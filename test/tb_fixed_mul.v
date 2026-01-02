// Testbench for fixed-point multiplier

`timescale 1ns/1ps

module tb_fixed_mul;

    reg  signed [15:0] a, b;
    wire signed [15:0] result;

    fixed_mul uut (
        .a(a),
        .b(b),
        .result(result)
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
        $display("=== Fixed-Point Multiplier Test (Q8.8) ===");
        $display("Format: a * b = result (expected)");
        $display("");

        // Test 1: 2.0 * 3.0 = 6.0
        a = real_to_q8(2.0);
        b = real_to_q8(3.0);
        #10;
        $display("  %.3f * %.3f = %.3f (expected 6.0)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        // Test 2: 1.5 * 2.0 = 3.0
        a = real_to_q8(1.5);
        b = real_to_q8(2.0);
        #10;
        $display("  %.3f * %.3f = %.3f (expected 3.0)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        // Test 3: 0.5 * 0.5 = 0.25
        a = real_to_q8(0.5);
        b = real_to_q8(0.5);
        #10;
        $display("  %.3f * %.3f = %.3f (expected 0.25)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        // Test 4: -1.0 * 2.0 = -2.0
        a = real_to_q8(-1.0);
        b = real_to_q8(2.0);
        #10;
        $display("  %.3f * %.3f = %.3f (expected -2.0)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        // Test 5: -0.5 * -0.5 = 0.25
        a = real_to_q8(-0.5);
        b = real_to_q8(-0.5);
        #10;
        $display("  %.3f * %.3f = %.3f (expected 0.25)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        // Test 6: Small numbers - 0.1 * 0.1 = 0.01 (precision test)
        a = real_to_q8(0.1);
        b = real_to_q8(0.1);
        #10;
        $display("  %.3f * %.3f = %.3f (expected ~0.01, precision limited)", 
                 q8_to_real(a), q8_to_real(b), q8_to_real(result));

        $display("");
        $display("=== Done ===");
        $finish;
    end

endmodule
