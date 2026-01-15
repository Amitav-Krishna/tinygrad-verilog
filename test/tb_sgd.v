module tb_sgd;
   // Network configuration: 2->2->2 (same as before)
   localparam NUM_LAYERS = 2;
   localparam [143:0] LAYER_SIZES = {16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd2, 16'd2, 16'd2};

   // Derived parameters
   localparam L0 = 2;
   localparam L1 = 2;
   localparam L2 = 2;
   localparam TOTAL_WEIGHTS = L1*L0 + L2*L1;  // 4 + 4 = 8
   localparam TOTAL_BIASES = L1 + L2;          // 2 + 2 = 4
   localparam TOTAL_ACTS = L0 + L1 + L2;       // 2 + 2 + 2 = 6

   // Clock and control
   reg clk;
   reg rst;
   reg start_fwd;
   reg start_bwd;

   // Network inputs
   reg signed [(L0*16)-1:0] x_vals;
   reg signed [(TOTAL_WEIGHTS*16)-1:0] w_vals;
   reg signed [(TOTAL_BIASES*16)-1:0] b_vals;

   // Network outputs
   wire signed [(L2*16)-1:0] y_vals;
   wire signed [(TOTAL_ACTS*16)-1:0] activations;
   wire fwd_done;

   // Target for loss computation
   reg signed [(L2*16)-1:0] target;

   // Loss gradient: dL/dy = y - target for MSE
   reg signed [(L2*16)-1:0] dL_dy;

   // Backward outputs
   wire signed [(TOTAL_WEIGHTS*16)-1:0] dL_dw;
   wire signed [(TOTAL_BIASES*16)-1:0] dL_db;
   wire bwd_done;

   // SGD inputs/outputs
   reg signed [15:0] lr;
   wire signed [(TOTAL_WEIGHTS*16)-1:0] w_new;
   wire signed [(TOTAL_BIASES*16)-1:0] b_new;

   // Instantiate network (forward pass)
   network #(
      .NUM_LAYERS(NUM_LAYERS),
      .LAYER_SIZES(LAYER_SIZES)
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
      .NUM_LAYERS(NUM_LAYERS),
      .LAYER_SIZES(LAYER_SIZES)
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
      .NUM_LAYERS(NUM_LAYERS),
      .LAYER_SIZES(LAYER_SIZES)
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
      input signed [(L0*16)-1:0] vec;
      integer idx;
      begin
         $write("%s: [", name);
         for (idx = 0; idx < L0; idx = idx + 1) begin
            $write("%.4f", q8_to_real(vec[idx*16 +: 16]));
            if (idx < L0-1) $write(", ");
         end
         $display("]");
      end
   endtask

   // Compute MSE loss
   function real compute_mse;
      input signed [(L2*16)-1:0] y;
      input signed [(L2*16)-1:0] t;
      real sum;
      real diff;
      integer idx;
      begin
         sum = 0.0;
         for (idx = 0; idx < L2; idx = idx + 1) begin
            diff = q8_to_real(y[idx*16 +: 16]) - q8_to_real(t[idx*16 +: 16]);
            sum = sum + diff * diff;
         end
         compute_mse = sum / 2.0;
      end
   endfunction

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

      // Set weights (8 total)
      w_vals = {
         // Layer 1 weights
         real_to_q8(0.2), real_to_q8(0.5),
         real_to_q8(0.3), real_to_q8(0.4),
         // Layer 0 weights
         real_to_q8(0.8), real_to_q8(0.1),
         real_to_q8(0.7), real_to_q8(0.9)
      };

      // Set biases (4 total)
      b_vals = {
         // Layer 1 biases
         real_to_q8(0.5),
         real_to_q8(1.0),
         // Layer 0 biases
         real_to_q8(0.6),
         real_to_q8(1.2)
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
