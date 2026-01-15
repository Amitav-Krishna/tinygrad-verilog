module tb_sgd;
   // Parameters
   localparam N = 2;
   localparam M = 2;

   // Clock and control
   reg clk;
   reg rst;
   reg start_fwd;
   reg start_bwd;

   // Network inputs
   reg signed [(N*16)-1:0] x_vals;
   reg signed [((M*(N*N))*16)-1:0] w_vals;
   reg signed [((N*M)*16)-1:0] b_vals;

   // Network outputs
   wire signed [(N*16)-1:0] y_vals;
   wire signed [(((M+1)*N)*16)-1:0] activations;
   wire fwd_done;

   // Target for loss computation
   reg signed [(N*16)-1:0] target;

   // Loss gradient: dL/dy = y - target for MSE
   reg signed [(N*16)-1:0] dL_dy;

   // Backward outputs
   wire signed [((M*(N*N))*16)-1:0] dL_dw;
   wire signed [((N*M)*16)-1:0] dL_db;
   wire bwd_done;

   // SGD inputs/outputs
   reg signed [15:0] lr;
   wire signed [((M*(N*N))*16)-1:0] w_new;
   wire signed [((N*M)*16)-1:0] b_new;

   // Instantiate network (forward pass)
   network #(
      .N(N),
      .M(M)
   ) net (
      .clk(clk),
      .start(start_fwd),
      .x(x_vals),
      .w(w_vals),
      .b(b_vals),
      .y(y_vals),
      .intermediate_states(activations),
      .done(fwd_done)
   );

   // Instantiate backward pass
   backward #(
      .N(N),
      .M(M)
   ) bwd (
      .clk(clk),
      .rst(rst),
      .start(start_bwd),
      .activations(activations),
      .w(w_vals),
      .dL_dy(dL_dy),
      .dL_dw(dL_dw),
      .dL_db(dL_db),
      .done(bwd_done)
   );

   // Instantiate SGD optimizer
   sgd #(
      .N(N),
      .M(M)
   ) optimizer (
      .w(w_vals),
      .b(b_vals),
      .dL_dw(dL_dw),
      .dL_db(dL_db),
      .lr(lr),
      .w_new(w_new),
      .b_new(b_new)
   );

   // Clock generation
   always #5 clk = ~clk;

   // Helper functions for Q8.8 fixed-point
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

   // Helper to display a vector
   task display_vector;
      input [8*20:1] name;
      input signed [(N*16)-1:0] vec;
      integer idx;
      begin
         $write("%s: [", name);
         for (idx = 0; idx < N; idx = idx + 1) begin
            $write("%.4f", q8_to_real(vec[idx*16 +: 16]));
            if (idx < N-1) $write(", ");
         end
         $display("]");
      end
   endtask

   // Helper to display weights for one layer
   task display_weights;
      input [8*10:1] name;
      input signed [((M*(N*N))*16)-1:0] weights;
      input integer layer;
      integer row, col;
      integer base;
      begin
         $display("  %s[%0d]:", name, layer);
         base = layer * N * N * 16;
         for (row = 0; row < N; row = row + 1) begin
            $write("    [");
            for (col = 0; col < N; col = col + 1) begin
               $write("%.4f", q8_to_real(weights[base + (row*N + col)*16 +: 16]));
               if (col < N-1) $write(", ");
            end
            $display("]");
         end
      end
   endtask

   // Helper to display biases for one layer
   task display_biases;
      input [8*10:1] name;
      input signed [((N*M)*16)-1:0] biases;
      input integer layer;
      integer idx;
      integer base;
      begin
         base = layer * N * 16;
         $write("  %s[%0d]: [", name, layer);
         for (idx = 0; idx < N; idx = idx + 1) begin
            $write("%.4f", q8_to_real(biases[base + idx*16 +: 16]));
            if (idx < N-1) $write(", ");
         end
         $display("]");
      end
   endtask

   // Compute MSE loss
   function real compute_mse;
      input signed [(N*16)-1:0] y;
      input signed [(N*16)-1:0] t;
      real sum;
      real diff;
      integer idx;
      begin
         sum = 0.0;
         for (idx = 0; idx < N; idx = idx + 1) begin
            diff = q8_to_real(y[idx*16 +: 16]) - q8_to_real(t[idx*16 +: 16]);
            sum = sum + diff * diff;
         end
         compute_mse = sum / 2.0;
      end
   endfunction

   integer layer_idx;
   real loss_before, loss_after;

   initial begin
      $display("=== SGD Optimizer Test ===");
      $display("");

      // Initialize
      clk = 0;
      rst = 1;
      start_fwd = 0;
      start_bwd = 0;

      // Set learning rate (0.1 in Q8.8)
      lr = real_to_q8(0.1);

      // Set input values
      x_vals = {
         real_to_q8(1.0),   // x[1]
         real_to_q8(2.0)    // x[0]
      };

      // Set weights
      w_vals = {
         // Layer 1: W[1]
         real_to_q8(0.2), real_to_q8(0.5),   // row 1
         real_to_q8(0.3), real_to_q8(0.4),   // row 0
         // Layer 0: W[0]
         real_to_q8(0.8), real_to_q8(0.1),   // row 1
         real_to_q8(0.7), real_to_q8(0.9)    // row 0
      };

      // Set biases
      b_vals = {
         // Layer 1: b[1]
         real_to_q8(0.5),   // b[1][1]
         real_to_q8(1.0),   // b[1][0]
         // Layer 0: b[0]
         real_to_q8(0.6),   // b[0][1]
         real_to_q8(1.2)    // b[0][0]
      };

      // Set target values
      target = {
         real_to_q8(1.0),   // target[1]
         real_to_q8(0.5)    // target[0]
      };

      // Release reset
      #20;
      rst = 0;
      #10;

      // ========== INITIAL PARAMETERS ==========
      $display("--- Initial Parameters ---");
      $display("Learning rate: %.4f", q8_to_real(lr));
      $display("");
      $display("Weights:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_weights("W", w_vals, layer_idx);
      end
      $display("Biases:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_biases("b", b_vals, layer_idx);
      end
      $display("");

      // ========== FORWARD PASS ==========
      $display("--- Forward Pass (Before SGD) ---");
      display_vector("Input x", x_vals);

      start_fwd = 1;
      #10;
      start_fwd = 0;

      wait(fwd_done);
      #10;

      display_vector("Output y", y_vals);
      display_vector("Target", target);
      loss_before = compute_mse(y_vals, target);
      $display("MSE Loss: %.6f", loss_before);
      $display("");

      // ========== COMPUTE LOSS GRADIENT ==========
      dL_dy[0*16 +: 16] = y_vals[0*16 +: 16] - target[0*16 +: 16];
      dL_dy[1*16 +: 16] = y_vals[1*16 +: 16] - target[1*16 +: 16];

      // ========== BACKWARD PASS ==========
      $display("--- Backward Pass ---");

      start_bwd = 1;
      #10;
      start_bwd = 0;

      wait(bwd_done);
      #10;

      $display("Gradients computed.");
      $display("");

      // ========== SGD UPDATE ==========
      $display("--- SGD Update ---");
      $display("w_new = w - lr * dL_dw");
      $display("b_new = b - lr * dL_db");
      $display("");

      // SGD is combinational, results are already available
      #10;

      $display("Updated Weights:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_weights("W_new", w_new, layer_idx);
      end
      $display("Updated Biases:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_biases("b_new", b_new, layer_idx);
      end
      $display("");

      // ========== APPLY UPDATED PARAMETERS ==========
      w_vals = w_new;
      b_vals = b_new;

      // ========== FORWARD PASS WITH NEW PARAMETERS ==========
      $display("--- Forward Pass (After SGD) ---");

      start_fwd = 1;
      #10;
      start_fwd = 0;

      wait(fwd_done);
      #10;

      display_vector("Output y", y_vals);
      display_vector("Target", target);
      loss_after = compute_mse(y_vals, target);
      $display("MSE Loss: %.6f", loss_after);
      $display("");

      // ========== VERIFY LOSS DECREASED ==========
      $display("--- Results ---");
      $display("Loss before: %.6f", loss_before);
      $display("Loss after:  %.6f", loss_after);
      if (loss_after < loss_before) begin
         $display("SUCCESS: Loss decreased by %.6f", loss_before - loss_after);
      end else begin
         $display("WARNING: Loss did not decrease!");
      end
      $display("");

      $display("=== Test Complete ===");
      $finish;
   end

endmodule
