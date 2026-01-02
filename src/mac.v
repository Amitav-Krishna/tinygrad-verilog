// Multiply-Accumulate (MAC) unit
// Computes: acc = sum(w[i] * x[i]) for i = 0 to N-1
//
// This is a SEQUENTIAL circuit - it uses a clock to accumulate over multiple cycles
// One multiply-add per clock cycle
//
// Protocol:
//   1. Assert 'start' and provide first (x, w) values
//   2. On each subsequent clock, provide next (x, w) values
//   3. After N clocks from start, 'done' goes high and 'acc' has result

module mac #(
    parameter N = 4  // Number of (x, w) pairs
)(
    input wire clk,
    input wire rst,              // Reset - clear accumulator
    input wire start,            // Start new computation (also loads first value)
    input wire signed [15:0] x,  // Input value (one per cycle)
    input wire signed [15:0] w,  // Weight value (one per cycle)
    output reg signed [15:0] acc,// Accumulated result
    output reg done              // High when computation complete
);

    // Counter to track how many values we've accumulated
    reg [$clog2(N+1)-1:0] count; 


    // Product from multiplier (combinational)
    wire signed [15:0] product;
    
    fixed_mul mul_inst (
        .a(x),
        .b(w),
        .result(product)
    );

    // Main state machine
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            acc <= 16'b0;
            count <= 0;
            done <= 0;
        end else if (start) begin
            // Start: load first product directly into accumulator
            acc <= product;
            count <= 1;
            done <= (N == 1);  // Edge case: single element
        end else if (count > 0 && count < N) begin
            // Continue accumulating
            acc <= acc + product;
            count <= count + 1;
            done <= (count == N - 1);
        end else if (done) begin
            // Hold result, clear done after one cycle
            done <= 0;
        end
    end

endmodule
