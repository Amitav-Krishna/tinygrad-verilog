// SGD optimizer with heterogeneous layer support
module sgd #(
    parameter MAX_LAYERS = 8,
    parameter NUM_LAYERS = 2,
    parameter [(MAX_LAYERS+1)*16-1:0] LAYER_SIZES = {16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd1, 16'd3, 16'd2}
) (
    input wire signed [(TOTAL_WEIGHTS*16)-1:0] w,
    input wire signed [(TOTAL_BIASES*16)-1:0] b,
    input wire signed [(TOTAL_WEIGHTS*16)-1:0] dL_dw,
    input wire signed [(TOTAL_BIASES*16)-1:0] dL_db,
    input wire signed [15:0] lr,
    output wire signed [(TOTAL_WEIGHTS*16)-1:0] w_new,
    output wire signed [(TOTAL_BIASES*16)-1:0] b_new
);

    // Extract layer sizes
    localparam L0 = LAYER_SIZES[0*16 +: 16];
    localparam L1 = LAYER_SIZES[1*16 +: 16];
    localparam L2 = LAYER_SIZES[2*16 +: 16];
    localparam L3 = LAYER_SIZES[3*16 +: 16];
    localparam L4 = LAYER_SIZES[4*16 +: 16];
    localparam L5 = LAYER_SIZES[5*16 +: 16];
    localparam L6 = LAYER_SIZES[6*16 +: 16];
    localparam L7 = LAYER_SIZES[7*16 +: 16];
    localparam L8 = LAYER_SIZES[8*16 +: 16];

    // Weight counts
    localparam W0 = L1 * L0;
    localparam W1 = L2 * L1;
    localparam W2 = L3 * L2;
    localparam W3 = L4 * L3;
    localparam W4 = L5 * L4;
    localparam W5 = L6 * L5;
    localparam W6 = L7 * L6;
    localparam W7 = L8 * L7;

    localparam TOTAL_WEIGHTS =
        ((NUM_LAYERS > 0) ? W0 : 0) + ((NUM_LAYERS > 1) ? W1 : 0) +
        ((NUM_LAYERS > 2) ? W2 : 0) + ((NUM_LAYERS > 3) ? W3 : 0) +
        ((NUM_LAYERS > 4) ? W4 : 0) + ((NUM_LAYERS > 5) ? W5 : 0) +
        ((NUM_LAYERS > 6) ? W6 : 0) + ((NUM_LAYERS > 7) ? W7 : 0);

    localparam TOTAL_BIASES =
        ((NUM_LAYERS > 0) ? L1 : 0) + ((NUM_LAYERS > 1) ? L2 : 0) +
        ((NUM_LAYERS > 2) ? L3 : 0) + ((NUM_LAYERS > 3) ? L4 : 0) +
        ((NUM_LAYERS > 4) ? L5 : 0) + ((NUM_LAYERS > 5) ? L6 : 0) +
        ((NUM_LAYERS > 6) ? L7 : 0) + ((NUM_LAYERS > 7) ? L8 : 0);

    localparam TOTAL_PARAMS = TOTAL_WEIGHTS + TOTAL_BIASES;

    // Concatenate params and gradients
    wire signed [(TOTAL_PARAMS*16)-1:0] params;
    wire signed [(TOTAL_PARAMS*16)-1:0] grads;
    wire signed [(TOTAL_PARAMS*16)-1:0] params_new;

    assign params = {w, b};
    assign grads = {dL_dw, dL_db};
    assign w_new = params_new[(TOTAL_BIASES*16) +: (TOTAL_WEIGHTS*16)];
    assign b_new = params_new[(TOTAL_BIASES*16)-1:0];

    // SGD update: param_new = param - lr * grad
    genvar i;
    generate
        for (i = 0; i < TOTAL_PARAMS; i = i + 1) begin : param_update
            wire signed [31:0] scaled_grad;
            // Cast part-selects to signed (part-selects lose signedness)
            assign scaled_grad = $signed(lr) * $signed(grads[i*16 +: 16]);
            assign params_new[i*16 +: 16] = $signed(params[i*16 +: 16]) - scaled_grad[23:8];
        end
    endgenerate

endmodule
