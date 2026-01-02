// Fixed-point Q8.8 multiplier
// Q8.8 = 8 integer bits, 8 fractional bits (16 bits total)
// Range: -128.0 to +127.996 (approximately)
// Resolution: 1/256 = 0.00390625
//
// Multiplication: Q8.8 * Q8.8 = Q16.16, then we extract middle 16 bits for Q8.8 result

module fixed_mul (
    input  signed [15:0] a,     // Q8.8 input
    input  signed [15:0] b,     // Q8.8 input
    output signed [15:0] result // Q8.8 output
);

    // Full precision multiply: 16-bit * 16-bit = 32-bit
    wire signed [31:0] full_product;
    assign full_product = a * b;

    // The product is Q16.16 (16 integer bits, 16 fractional bits)
    // We want Q8.8, so we take bits [23:8]
    // This drops the top 8 bits (overflow risk) and bottom 8 bits (precision loss)
    assign result = full_product[23:8];

endmodule
