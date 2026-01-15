// Neural network with support for heterogeneous layer sizes
// Layer sizes passed as packed parameter

module network #(
    parameter MAX_LAYERS = 8,
    parameter NUM_LAYERS = 2,
    // Packed layer sizes: {L_MAX, ..., L2, L1, L0} each 16 bits
    // Example: 2->3->1 is {16'd1, 16'd3, 16'd2}
    parameter [(MAX_LAYERS+1)*16-1:0] LAYER_SIZES = {16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd1, 16'd3, 16'd2}
) (
    input wire clk,
    input wire start,
    input wire signed [(L0*16)-1:0] x,
    input wire signed [(TOTAL_WEIGHTS*16)-1:0] w,
    input wire signed [(TOTAL_BIASES*16)-1:0] b,
    output wire signed [(L_OUT*16)-1:0] y,
    output wire signed [(TOTAL_ACTS*16)-1:0] intermediate_states,
    output wire done
);

    // Extract individual layer sizes as localparams
    localparam L0 = LAYER_SIZES[0*16 +: 16];
    localparam L1 = LAYER_SIZES[1*16 +: 16];
    localparam L2 = LAYER_SIZES[2*16 +: 16];
    localparam L3 = LAYER_SIZES[3*16 +: 16];
    localparam L4 = LAYER_SIZES[4*16 +: 16];
    localparam L5 = LAYER_SIZES[5*16 +: 16];
    localparam L6 = LAYER_SIZES[6*16 +: 16];
    localparam L7 = LAYER_SIZES[7*16 +: 16];
    localparam L8 = LAYER_SIZES[8*16 +: 16];

    // Output size
    localparam L_OUT = (NUM_LAYERS == 1) ? L1 :
                       (NUM_LAYERS == 2) ? L2 :
                       (NUM_LAYERS == 3) ? L3 :
                       (NUM_LAYERS == 4) ? L4 :
                       (NUM_LAYERS == 5) ? L5 :
                       (NUM_LAYERS == 6) ? L6 :
                       (NUM_LAYERS == 7) ? L7 :
                       (NUM_LAYERS == 8) ? L8 : 0;

    // Weight counts per layer
    localparam W0 = L1 * L0;
    localparam W1 = L2 * L1;
    localparam W2 = L3 * L2;
    localparam W3 = L4 * L3;
    localparam W4 = L5 * L4;
    localparam W5 = L6 * L5;
    localparam W6 = L7 * L6;
    localparam W7 = L8 * L7;

    // Total weights
    localparam TOTAL_WEIGHTS =
        ((NUM_LAYERS > 0) ? W0 : 0) +
        ((NUM_LAYERS > 1) ? W1 : 0) +
        ((NUM_LAYERS > 2) ? W2 : 0) +
        ((NUM_LAYERS > 3) ? W3 : 0) +
        ((NUM_LAYERS > 4) ? W4 : 0) +
        ((NUM_LAYERS > 5) ? W5 : 0) +
        ((NUM_LAYERS > 6) ? W6 : 0) +
        ((NUM_LAYERS > 7) ? W7 : 0);

    // Weight offsets
    localparam WO0 = 0;
    localparam WO1 = W0;
    localparam WO2 = W0 + W1;
    localparam WO3 = W0 + W1 + W2;
    localparam WO4 = W0 + W1 + W2 + W3;
    localparam WO5 = W0 + W1 + W2 + W3 + W4;
    localparam WO6 = W0 + W1 + W2 + W3 + W4 + W5;
    localparam WO7 = W0 + W1 + W2 + W3 + W4 + W5 + W6;

    // Total biases
    localparam TOTAL_BIASES =
        ((NUM_LAYERS > 0) ? L1 : 0) +
        ((NUM_LAYERS > 1) ? L2 : 0) +
        ((NUM_LAYERS > 2) ? L3 : 0) +
        ((NUM_LAYERS > 3) ? L4 : 0) +
        ((NUM_LAYERS > 4) ? L5 : 0) +
        ((NUM_LAYERS > 5) ? L6 : 0) +
        ((NUM_LAYERS > 6) ? L7 : 0) +
        ((NUM_LAYERS > 7) ? L8 : 0);

    // Bias offsets
    localparam BO0 = 0;
    localparam BO1 = L1;
    localparam BO2 = L1 + L2;
    localparam BO3 = L1 + L2 + L3;
    localparam BO4 = L1 + L2 + L3 + L4;
    localparam BO5 = L1 + L2 + L3 + L4 + L5;
    localparam BO6 = L1 + L2 + L3 + L4 + L5 + L6;
    localparam BO7 = L1 + L2 + L3 + L4 + L5 + L6 + L7;

    // Total activations
    localparam TOTAL_ACTS = L0 +
        ((NUM_LAYERS > 0) ? L1 : 0) +
        ((NUM_LAYERS > 1) ? L2 : 0) +
        ((NUM_LAYERS > 2) ? L3 : 0) +
        ((NUM_LAYERS > 3) ? L4 : 0) +
        ((NUM_LAYERS > 4) ? L5 : 0) +
        ((NUM_LAYERS > 5) ? L6 : 0) +
        ((NUM_LAYERS > 6) ? L7 : 0) +
        ((NUM_LAYERS > 7) ? L8 : 0);

    // Activation offsets
    localparam AO0 = 0;
    localparam AO1 = L0;
    localparam AO2 = L0 + L1;
    localparam AO3 = L0 + L1 + L2;
    localparam AO4 = L0 + L1 + L2 + L3;
    localparam AO5 = L0 + L1 + L2 + L3 + L4;
    localparam AO6 = L0 + L1 + L2 + L3 + L4 + L5;
    localparam AO7 = L0 + L1 + L2 + L3 + L4 + L5 + L6;
    localparam AO8 = L0 + L1 + L2 + L3 + L4 + L5 + L6 + L7;

    // Macros for indexed access (i must be constant)
    `define GET_L(i) ((i)==0 ? L0 : (i)==1 ? L1 : (i)==2 ? L2 : (i)==3 ? L3 : (i)==4 ? L4 : (i)==5 ? L5 : (i)==6 ? L6 : (i)==7 ? L7 : L8)
    `define GET_WO(i) ((i)==0 ? WO0 : (i)==1 ? WO1 : (i)==2 ? WO2 : (i)==3 ? WO3 : (i)==4 ? WO4 : (i)==5 ? WO5 : (i)==6 ? WO6 : WO7)
    `define GET_BO(i) ((i)==0 ? BO0 : (i)==1 ? BO1 : (i)==2 ? BO2 : (i)==3 ? BO3 : (i)==4 ? BO4 : (i)==5 ? BO5 : (i)==6 ? BO6 : BO7)
    `define GET_AO(i) ((i)==0 ? AO0 : (i)==1 ? AO1 : (i)==2 ? AO2 : (i)==3 ? AO3 : (i)==4 ? AO4 : (i)==5 ? AO5 : (i)==6 ? AO6 : (i)==7 ? AO7 : AO8)

    // Layer signals
    wire [NUM_LAYERS-1:0] layer_done;
    wire [NUM_LAYERS-1:0] layer_start;

    // Copy input to intermediate_states
    assign intermediate_states[0 +: (L0*16)] = x;

    // Chain start signals
    assign layer_start[0] = start;
    genvar s;
    generate
        for (s = 1; s < NUM_LAYERS; s = s + 1) begin : start_chain
            assign layer_start[s] = layer_done[s-1];
        end
    endgenerate

    // Output and done
    assign y = intermediate_states[(`GET_AO(NUM_LAYERS)*16) +: (L_OUT*16)];
    assign done = layer_done[NUM_LAYERS-1];

    // Generate layers
    genvar i;
    generate
        for (i = 0; i < NUM_LAYERS; i = i + 1) begin : layers
            layer #(
                .N_IN(`GET_L(i)),
                .N_OUT(`GET_L(i+1))
            ) layer_inst (
                .clk(clk),
                .start(layer_start[i]),
                .x(intermediate_states[(`GET_AO(i)*16) +: (`GET_L(i)*16)]),
                .w(w[(`GET_WO(i)*16) +: (`GET_L(i+1)*`GET_L(i)*16)]),
                .b(b[(`GET_BO(i)*16) +: (`GET_L(i+1)*16)]),
                .y(intermediate_states[(`GET_AO(i+1)*16) +: (`GET_L(i+1)*16)]),
                .done(layer_done[i])
            );
        end
    endgenerate

endmodule
