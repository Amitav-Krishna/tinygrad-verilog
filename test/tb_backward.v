module tb_backward;
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

   // Loss gradient: dL/dy = 2 * (y - target) for MSE
   reg signed [(N*16)-1:0] dL_dy;

   // Backward outputs
   wire signed [((M*(N*N))*16)-1:0] dL_dw;
   wire signed [((N*M)*16)-1:0] dL_db;
   wire bwd_done;

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

   // Helper to display weight gradients for one layer
   task display_weight_grad;
      input integer layer;
      integer row, col;
      integer base;
      begin
         $display("  dL_dW[%0d]:", layer);
         base = layer * N * N * 16;
         for (row = 0; row < N; row = row + 1) begin
            $write("    [");
            for (col = 0; col < N; col = col + 1) begin
               $write("%.4f", q8_to_real(dL_dw[base + (row*N + col)*16 +: 16]));
               if (col < N-1) $write(", ");
            end
            $display("]");
         end
      end
   endtask

   // Helper to display bias gradients for one layer
   task display_bias_grad;
      input integer layer;
      integer idx;
      integer base;
      begin
         base = layer * N * 16;
         $write("  dL_db[%0d]: [", layer);
         for (idx = 0; idx < N; idx = idx + 1) begin
            $write("%.4f", q8_to_real(dL_db[base + idx*16 +: 16]));
            if (idx < N-1) $write(", ");
         end
         $display("]");
      end
   endtask

   integer layer_idx;

   initial begin
      $display("=== Forward + Backward Pass Test ===");
      $display("");

      // Initialize
      clk = 0;
      rst = 1;
      start_fwd = 0;
      start_bwd = 0;

      // Set input values
      x_vals = {
         real_to_q8(1.0),   // x[1]
         real_to_q8(2.0)    // x[0]
      };

      // Set weights (same as tb_network)
      // Layer 0 weights, then Layer 1 weights
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

      // ========== FORWARD PASS ==========
      $display("--- Forward Pass ---");
      display_vector("Input x", x_vals);
      $display("");

      start_fwd = 1;
      #10;
      start_fwd = 0;

      // Wait for forward pass to complete
      wait(fwd_done);
      #10;

      display_vector("Output y", y_vals);
      display_vector("Target", target);
      $display("");

      // Display intermediate activations
      $display("Intermediate activations:");
      $write("  a[0] (input): [");
      $write("%.4f, ", q8_to_real(activations[0*16 +: 16]));
      $write("%.4f", q8_to_real(activations[1*16 +: 16]));
      $display("]");
      $write("  a[1] (layer0 out): [");
      $write("%.4f, ", q8_to_real(activations[2*16 +: 16]));
      $write("%.4f", q8_to_real(activations[3*16 +: 16]));
      $display("]");
      $write("  a[2] (layer1 out): [");
      $write("%.4f, ", q8_to_real(activations[4*16 +: 16]));
      $write("%.4f", q8_to_real(activations[5*16 +: 16]));
      $display("]");
      $display("");

      // ========== COMPUTE LOSS GRADIENT ==========
      // MSE loss: L = (1/2) * sum((y - target)^2)
      // dL/dy = y - target (simplified, factor of 2 cancels with 1/2)
      $display("--- Computing Loss Gradient ---");
      dL_dy[0*16 +: 16] = y_vals[0*16 +: 16] - target[0*16 +: 16];
      dL_dy[1*16 +: 16] = y_vals[1*16 +: 16] - target[1*16 +: 16];
      display_vector("dL/dy", dL_dy);
      $display("");

      // ========== BACKWARD PASS ==========
      $display("--- Backward Pass ---");

      start_bwd = 1;
      #10;
      start_bwd = 0;

      // Wait for backward pass to complete
      wait(bwd_done);
      #10;

      $display("Gradients computed:");
      $display("");

      // Display weight gradients
      $display("Weight gradients:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_weight_grad(layer_idx);
      end
      $display("");

      // Display bias gradients
      $display("Bias gradients:");
      for (layer_idx = 0; layer_idx < M; layer_idx = layer_idx + 1) begin
         display_bias_grad(layer_idx);
      end
      $display("");

      $display("=== Test Complete ===");
      $finish;
   end

endmodule
