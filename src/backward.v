module backward #(
		  parameter N = 4,
		  parameter M = 4
		  ) (
		     input wire clk,
		     input wire rst,
		     input wire start,
		     input wire signed [(((M+1)*N)*16)-1:0] activations,
		     input wire signed [((M*(N*N))*16)-1:0] w,
		     input wire signed [(N*16)-1:0] dL_dy,
		     output reg signed [((M*(N*N))*16)-1:0] dL_dw,
		     output reg signed [((N*M)*16)-1:0] dL_db,
		     output reg done
		     );

   // State encoding
   localparam IDLE        = 4'd0;
   localparam INIT_DZ     = 4'd1;  // initialize counter for COMPUTE_DZ
   localparam COMPUTE_DZ  = 4'd2;  // dL_dz = dL_da ⊙ ReLU'(act)
   localparam INIT_DW     = 4'd3;  // initialize counters for COMPUTE_DW
   localparam COMPUTE_DW  = 4'd4;  // dL_dW = dL_dz × a_prev^T
   localparam INIT_DA     = 4'd5;  // initialize counters for COMPUTE_DA
   localparam COMPUTE_DA  = 4'd6;  // dL_da_prev = W^T × dL_dz
   localparam STORE_DA    = 4'd7;  // store accumulated result
   localparam NEXT_LAYER  = 4'd8;  // decrement layer, loop or finish
   localparam DONE_STATE  = 4'd9;

   reg [3:0]  state;
   reg [$clog2(M)-1:0] layer_idx;  // current layer: M-1 down to 0

   // Working registers
   reg signed [(N*16)-1:0] dL_da;      // gradient w.r.t. current layer activation
   reg signed [(N*16)-1:0] dL_dz;      // gradient w.r.t. pre-activation
   reg signed [(N*16)-1:0] dL_da_prev; // gradient to pass to next iteration

   // For matrix/vector operations within each state
   reg [$clog2(N):0]	   i, j;  // extra bit to handle N as terminal value

   // Helper wires for indexing into activations
   // activations layout: [input (N)][layer0 out (N)][layer1 out (N)]...[layerM-1 out (N)]
   // layer_idx output is at activations[(layer_idx+1)*N + i]
   wire [31:0]		   act_idx;
   wire signed [15:0]	   act_val;
   assign act_idx = ((layer_idx + 1) * N + i) * 16;
   assign act_val = activations[act_idx +: 16];

   // Helper for dL_da element access
   wire signed [15:0] dL_da_i;
   assign dL_da_i = dL_da[i*16 +: 16];

   // Helper for dL_db indexing: dL_db layout is [layer0 (N)][layer1 (N)]...[layerM-1 (N)]
   wire [31:0] db_idx;
   assign db_idx = (layer_idx * N + i) * 16;

   // Helper for a_prev (input to current layer) = activations[layer_idx]
   // a_prev[j] is at activations[layer_idx * N + j]
   wire [31:0] a_prev_idx;
   wire signed [15:0] a_prev_j;
   assign a_prev_idx = (layer_idx * N + j) * 16;
   assign a_prev_j = activations[a_prev_idx +: 16];

   // Helper for dL_dz[i] access
   wire signed [15:0] dL_dz_i;
   assign dL_dz_i = dL_dz[i*16 +: 16];

   // Helper for dL_dw indexing: dL_dw layout is [layer0 (N×N)][layer1 (N×N)]...
   // Weight [i][j] of layer_idx is at: (layer_idx * N * N + i * N + j)
   wire [31:0] dw_idx;
   assign dw_idx = (layer_idx * N * N + i * N + j) * 16;

   // Fixed-point multiplier result (Q8.8 × Q8.8 = Q16.16, take middle 16 bits)
   wire signed [31:0] mult_result;
   assign mult_result = dL_dz_i * a_prev_j;
   wire signed [15:0] mult_q8_8;
   assign mult_q8_8 = mult_result[23:8];  // extract Q8.8 from Q16.16

   // Helper for W[layer_idx][i][j] access (same indexing as dw_idx)
   // W[i][j] is at: (layer_idx * N * N + i * N + j)
   wire [31:0] w_idx;
   wire signed [15:0] w_ij;
   assign w_idx = (layer_idx * N * N + i * N + j) * 16;
   assign w_ij = w[w_idx +: 16];

   // Multiplier for W^T × dL_dz: W[i][j] * dL_dz[i]
   wire signed [31:0] mult_da_result;
   assign mult_da_result = w_ij * dL_dz_i;
   wire signed [15:0] mult_da_q8_8;
   assign mult_da_q8_8 = mult_da_result[23:8];

   // Accumulator for computing dL_da_prev[j] = sum_i(W[i][j] * dL_dz[i])
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
                 layer_idx <= M - 1;
                 dL_da <= dL_dy;  // start with loss gradient
                 state <= INIT_DZ;
              end
           end

           INIT_DZ: begin
              // Initialize counter for COMPUTE_DZ loop
              i <= 0;
              state <= COMPUTE_DZ;
           end

           COMPUTE_DZ: begin
              // dL_dz[i] = dL_da[i] * ReLU'(activation)
              // ReLU'(x) = 1 if x > 0, else 0
              // Since activation is post-ReLU, check if act > 0
              if (act_val > 0) begin
                 dL_dz[i*16 +: 16] <= dL_da_i;
              end else begin
                 dL_dz[i*16 +: 16] <= 16'sd0;
              end
              // Also store to dL_db (bias gradient = dL_dz)
              if (act_val > 0) begin
                 dL_db[db_idx +: 16] <= dL_da_i;
              end else begin
                 dL_db[db_idx +: 16] <= 16'sd0;
              end

              // Increment or move to next state
              if (i == N - 1) begin
                 state <= INIT_DW;
              end else begin
                 i <= i + 1;
              end
           end

           INIT_DW: begin
              // Initialize counters for nested loop
              i <= 0;
              j <= 0;
              state <= COMPUTE_DW;
           end

           COMPUTE_DW: begin
              // dL_dW[layer_idx][i][j] = dL_dz[i] * a_prev[j]
              // Outer product: N×N operations
              dL_dw[dw_idx +: 16] <= mult_q8_8;

              // Nested loop: iterate j, then i
              if (j == N - 1) begin
                 j <= 0;
                 if (i == N - 1) begin
                    state <= INIT_DA;
                 end else begin
                    i <= i + 1;
                 end
              end else begin
                 j <= j + 1;
              end
           end

           INIT_DA: begin
              // Initialize for dL_da_prev computation
              // Outer loop over j (output index), inner loop over i (sum index)
              i <= 0;
              j <= 0;
              acc <= 16'sd0;
              dL_da_prev <= 0;
              state <= COMPUTE_DA;
           end

           COMPUTE_DA: begin
              // dL_da_prev[j] = sum_i(W[i][j] * dL_dz[i])
              // Accumulate W[i][j] * dL_dz[i] for current i
              acc <= acc + mult_da_q8_8;

              // Inner loop over i
              if (i == N - 1) begin
                 // Done summing for this j, go store result
                 state <= STORE_DA;
              end else begin
                 i <= i + 1;
              end
           end

           STORE_DA: begin
              // Store accumulated result to dL_da_prev[j]
              dL_da_prev[j*16 +: 16] <= acc;

              // Move to next j or finish
              if (j == N - 1) begin
                 state <= NEXT_LAYER;
              end else begin
                 j <= j + 1;
                 i <= 0;
                 acc <= 16'sd0;
                 state <= COMPUTE_DA;
              end
           end

           NEXT_LAYER: begin
              dL_da <= dL_da_prev;  // pass gradient backward
              if (layer_idx == 0) begin
                 state <= DONE_STATE;
              end else begin
                 layer_idx <= layer_idx - 1;
                 state <= INIT_DZ;  // reinitialize counter for next layer
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
