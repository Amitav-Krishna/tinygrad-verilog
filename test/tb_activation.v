
`timescale 1ns/1ps  // time unit / precision

module tb_activation;

    // Test signals
    reg signed [15:0] weight_product;    // reg = can be assigned in procedural blocks
    reg signed [15:0] b;
    wire [15:0] activation;     // wire = driven by continuous assignment

    // Instantiate the module we're testing
    activation uut (         // uut = "unit under test"
        .weight_product(weight_product),
        .b(b),
        .activation(activation)
    );

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

    // Test procedure
    initial begin
        // $display is like printf
        $display("=== Activation Testbench ===");
        
        // Test case 1
        weight_product = real_to_q8(10);  // 'd = decimal
        b = real_to_q8(20);
        #10;        // wait 10 time units (let signals propagate)
        $display("The activation from %.3f as the weight-input dot product and %.3f as the bias is %.3f", q8_to_real(weight_product), q8_to_real(b), q8_to_real(activation));

        // Test case 2
        weight_product = real_to_q8(-10);  // 'd = decimal
        b = real_to_q8(-20);
        #10;        // wait 10 time units (let signals propagate)
        $display("The activation from %.3f as the weight-input dot product and %.3f as the bias is %.3f", q8_to_real(weight_product), q8_to_real(b), q8_to_real(activation));


    end

endmodule
