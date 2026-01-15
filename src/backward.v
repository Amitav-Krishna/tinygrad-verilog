// Backward pass with heterogeneous layer support
module backward #(
    parameter MAX_LAYERS = 8,
    parameter NUM_LAYERS = 2,
    parameter [(MAX_LAYERS+1)*16-1:0] LAYER_SIZES = {16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd1, 16'd3, 16'd2}
) (
    input wire clk,
    input wire rst,
    input wire start,
    input wire signed [(TOTAL_ACTS*16)-1:0] activations,
    input wire signed [(TOTAL_WEIGHTS*16)-1:0] w,
    input wire signed [(L_OUT*16)-1:0] dL_dy,
    output reg signed [(TOTAL_WEIGHTS*16)-1:0] dL_dw,
    output reg signed [(TOTAL_BIASES*16)-1:0] dL_db,
    output reg done
);

    // Extract layer sizes as localparams
    localparam L0 = LAYER_SIZES[0*16 +: 16];
    localparam L1 = LAYER_SIZES[1*16 +: 16];
    localparam L2 = LAYER_SIZES[2*16 +: 16];
    localparam L3 = LAYER_SIZES[3*16 +: 16];
    localparam L4 = LAYER_SIZES[4*16 +: 16];
    localparam L5 = LAYER_SIZES[5*16 +: 16];
    localparam L6 = LAYER_SIZES[6*16 +: 16];
    localparam L7 = LAYER_SIZES[7*16 +: 16];
    localparam L8 = LAYER_SIZES[8*16 +: 16];

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

    localparam TOTAL_WEIGHTS =
        ((NUM_LAYERS > 0) ? W0 : 0) + ((NUM_LAYERS > 1) ? W1 : 0) +
        ((NUM_LAYERS > 2) ? W2 : 0) + ((NUM_LAYERS > 3) ? W3 : 0) +
        ((NUM_LAYERS > 4) ? W4 : 0) + ((NUM_LAYERS > 5) ? W5 : 0) +
        ((NUM_LAYERS > 6) ? W6 : 0) + ((NUM_LAYERS > 7) ? W7 : 0);

    localparam WO0 = 0;
    localparam WO1 = W0;
    localparam WO2 = W0 + W1;
    localparam WO3 = W0 + W1 + W2;
    localparam WO4 = W0 + W1 + W2 + W3;
    localparam WO5 = W0 + W1 + W2 + W3 + W4;
    localparam WO6 = W0 + W1 + W2 + W3 + W4 + W5;
    localparam WO7 = W0 + W1 + W2 + W3 + W4 + W5 + W6;

    localparam TOTAL_BIASES =
        ((NUM_LAYERS > 0) ? L1 : 0) + ((NUM_LAYERS > 1) ? L2 : 0) +
        ((NUM_LAYERS > 2) ? L3 : 0) + ((NUM_LAYERS > 3) ? L4 : 0) +
        ((NUM_LAYERS > 4) ? L5 : 0) + ((NUM_LAYERS > 5) ? L6 : 0) +
        ((NUM_LAYERS > 6) ? L7 : 0) + ((NUM_LAYERS > 7) ? L8 : 0);

    localparam BO0 = 0;
    localparam BO1 = L1;
    localparam BO2 = L1 + L2;
    localparam BO3 = L1 + L2 + L3;
    localparam BO4 = L1 + L2 + L3 + L4;
    localparam BO5 = L1 + L2 + L3 + L4 + L5;
    localparam BO6 = L1 + L2 + L3 + L4 + L5 + L6;
    localparam BO7 = L1 + L2 + L3 + L4 + L5 + L6 + L7;

    localparam TOTAL_ACTS = L0 +
        ((NUM_LAYERS > 0) ? L1 : 0) + ((NUM_LAYERS > 1) ? L2 : 0) +
        ((NUM_LAYERS > 2) ? L3 : 0) + ((NUM_LAYERS > 3) ? L4 : 0) +
        ((NUM_LAYERS > 4) ? L5 : 0) + ((NUM_LAYERS > 5) ? L6 : 0) +
        ((NUM_LAYERS > 6) ? L7 : 0) + ((NUM_LAYERS > 7) ? L8 : 0);

    localparam AO0 = 0;
    localparam AO1 = L0;
    localparam AO2 = L0 + L1;
    localparam AO3 = L0 + L1 + L2;
    localparam AO4 = L0 + L1 + L2 + L3;
    localparam AO5 = L0 + L1 + L2 + L3 + L4;
    localparam AO6 = L0 + L1 + L2 + L3 + L4 + L5;
    localparam AO7 = L0 + L1 + L2 + L3 + L4 + L5 + L6;
    localparam AO8 = L0 + L1 + L2 + L3 + L4 + L5 + L6 + L7;

    // Max layer size for register sizing
    localparam MAX_L = (L0 > L1 ? L0 : L1) > (L2 > L3 ? L2 : L3) ?
                       (L0 > L1 ? L0 : L1) : (L2 > L3 ? L2 : L3);

    // Runtime lookup functions (synthesizable)
    function [15:0] get_L;
        input [3:0] idx;
        begin
            case (idx)
                0: get_L = L0; 1: get_L = L1; 2: get_L = L2; 3: get_L = L3;
                4: get_L = L4; 5: get_L = L5; 6: get_L = L6; 7: get_L = L7;
                default: get_L = L8;
            endcase
        end
    endfunction

    function [31:0] get_WO;
        input [3:0] idx;
        begin
            case (idx)
                0: get_WO = WO0; 1: get_WO = WO1; 2: get_WO = WO2; 3: get_WO = WO3;
                4: get_WO = WO4; 5: get_WO = WO5; 6: get_WO = WO6; default: get_WO = WO7;
            endcase
        end
    endfunction

    function [31:0] get_BO;
        input [3:0] idx;
        begin
            case (idx)
                0: get_BO = BO0; 1: get_BO = BO1; 2: get_BO = BO2; 3: get_BO = BO3;
                4: get_BO = BO4; 5: get_BO = BO5; 6: get_BO = BO6; default: get_BO = BO7;
            endcase
        end
    endfunction

    function [31:0] get_AO;
        input [3:0] idx;
        begin
            case (idx)
                0: get_AO = AO0; 1: get_AO = AO1; 2: get_AO = AO2; 3: get_AO = AO3;
                4: get_AO = AO4; 5: get_AO = AO5; 6: get_AO = AO6; 7: get_AO = AO7;
                default: get_AO = AO8;
            endcase
        end
    endfunction

    // State encoding
    localparam IDLE        = 4'd0;
    localparam INIT_DZ     = 4'd1;
    localparam COMPUTE_DZ  = 4'd2;
    localparam INIT_DW     = 4'd3;
    localparam COMPUTE_DW  = 4'd4;
    localparam INIT_DA     = 4'd5;
    localparam COMPUTE_DA  = 4'd6;
    localparam STORE_DA    = 4'd7;
    localparam NEXT_LAYER  = 4'd8;
    localparam DONE_STATE  = 4'd9;

    reg [3:0] state;
    reg [3:0] layer_idx;

    // Current layer dimensions (set at start of each layer)
    reg [15:0] n_out;      // output neurons of current layer
    reg [15:0] n_in;       // input neurons of current layer
    reg [31:0] w_offset;   // weight offset for current layer
    reg [31:0] b_offset;   // bias offset for current layer
    reg [31:0] act_offset; // activation offset for current layer output
    reg [31:0] act_in_offset; // activation offset for current layer input

    // Working registers (sized for max layer)
    reg signed [(MAX_L*16)-1:0] dL_da;
    reg signed [(MAX_L*16)-1:0] dL_dz;
    reg signed [(MAX_L*16)-1:0] dL_da_prev;

    // Loop counters
    reg [15:0] i, j;

    // Element access
    wire signed [15:0] act_val;
    assign act_val = activations[(act_offset + i) * 16 +: 16];

    wire signed [15:0] dL_da_i;
    assign dL_da_i = dL_da[i*16 +: 16];

    wire signed [15:0] dL_dz_i;
    assign dL_dz_i = dL_dz[i*16 +: 16];

    wire signed [15:0] a_prev_j;
    assign a_prev_j = activations[(act_in_offset + j) * 16 +: 16];

    wire signed [15:0] w_ij;
    assign w_ij = w[(w_offset + i * n_in + j) * 16 +: 16];

    // Multipliers
    wire signed [31:0] mult_dw;
    assign mult_dw = dL_dz_i * a_prev_j;

    wire signed [31:0] mult_da;
    assign mult_da = w_ij * dL_dz_i;

    // Accumulator
    reg signed [15:0] acc;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state <= IDLE;
            done <= 0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 0;
                    if (start) begin
                        layer_idx <= NUM_LAYERS - 1;
                        dL_da[(L_OUT*16)-1:0] <= dL_dy;
                        state <= INIT_DZ;
                    end
                end

                INIT_DZ: begin
                    // Set dimensions for current layer
                    n_out <= get_L(layer_idx + 1);
                    n_in <= get_L(layer_idx);
                    w_offset <= get_WO(layer_idx);
                    b_offset <= get_BO(layer_idx);
                    act_offset <= get_AO(layer_idx + 1);
                    act_in_offset <= get_AO(layer_idx);
                    i <= 0;
                    state <= COMPUTE_DZ;
                end

                COMPUTE_DZ: begin
                    // dL_dz[i] = dL_da[i] * ReLU'(activation)
                    if (act_val > 0) begin
                        dL_dz[i*16 +: 16] <= dL_da_i;
                        dL_db[(b_offset + i) * 16 +: 16] <= dL_da_i;
                    end else begin
                        dL_dz[i*16 +: 16] <= 16'sd0;
                        dL_db[(b_offset + i) * 16 +: 16] <= 16'sd0;
                    end

                    if (i == n_out - 1) begin
                        state <= INIT_DW;
                    end else begin
                        i <= i + 1;
                    end
                end

                INIT_DW: begin
                    i <= 0;
                    j <= 0;
                    state <= COMPUTE_DW;
                end

                COMPUTE_DW: begin
                    // dL_dW[i][j] = dL_dz[i] * a_prev[j]
                    dL_dw[(w_offset + i * n_in + j) * 16 +: 16] <= mult_dw[23:8];

                    if (j == n_in - 1) begin
                        j <= 0;
                        if (i == n_out - 1) begin
                            state <= INIT_DA;
                        end else begin
                            i <= i + 1;
                        end
                    end else begin
                        j <= j + 1;
                    end
                end

                INIT_DA: begin
                    i <= 0;
                    j <= 0;
                    acc <= 16'sd0;
                    dL_da_prev <= 0;
                    state <= COMPUTE_DA;
                end

                COMPUTE_DA: begin
                    // dL_da_prev[j] = sum_i(W[i][j] * dL_dz[i])
                    acc <= acc + mult_da[23:8];

                    if (i == n_out - 1) begin
                        state <= STORE_DA;
                    end else begin
                        i <= i + 1;
                    end
                end

                STORE_DA: begin
                    dL_da_prev[j*16 +: 16] <= acc;

                    if (j == n_in - 1) begin
                        state <= NEXT_LAYER;
                    end else begin
                        j <= j + 1;
                        i <= 0;
                        acc <= 16'sd0;
                        state <= COMPUTE_DA;
                    end
                end

                NEXT_LAYER: begin
                    dL_da <= dL_da_prev;
                    if (layer_idx == 0) begin
                        state <= DONE_STATE;
                    end else begin
                        layer_idx <= layer_idx - 1;
                        state <= INIT_DZ;
                    end
                end

                DONE_STATE: begin
                    done <= 1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule
