// Testbench for adder
// Testbenches are simulation-only code - they don't synthesize to hardware

`timescale 1ns/1ps  // time unit / precision

module tb_adder;

    // Test signals
    reg  [7:0] a, b;    // reg = can be assigned in procedural blocks
    wire [8:0] sum;     // wire = driven by continuous assignment

    // Instantiate the module we're testing
    adder uut (         // uut = "unit under test"
        .a(a),
        .b(b),
        .sum(sum)
    );

    // Test procedure
    initial begin
        // $display is like printf
        $display("=== Adder Testbench ===");
        
        // Test case 1
        a = 8'd10;  // 'd = decimal
        b = 8'd20;
        #10;        // wait 10 time units (let signals propagate)
        $display("  %d + %d = %d", a, b, sum);

        // Test case 2
        a = 8'd255;
        b = 8'd1;
        #10;
        $display("  %d + %d = %d (overflow test)", a, b, sum);

        // Test case 3
        a = 8'd100;
        b = 8'd100;
        #10;
        $display("  %d + %d = %d", a, b, sum);

        $display("=== Done ===");
        $finish;
    end

endmodule
