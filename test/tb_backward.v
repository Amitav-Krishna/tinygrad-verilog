module tb_backward;
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

   // Loss gradient: dL/dy = 2 * (y - target) for MSE
   reg signed [(L2*16)-1:0] dL_dy;

   // Backward outputs
   wire signed [(TOTAL_WEIGHTS*16)-1:0] dL_dw;
   wire signed [(TOTAL_BIASES*16)-1:0] dL_db;
   wire bwd_done;

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

      // Set weights (8 total: layer0 has 4, layer1 has 4)
      // Layer 0: 2 neurons x 2 inputs = 4 weights
      // Layer 1: 2 neurons x 2 inputs = 4 weights
      w_vals = {
         // Layer 1 weights (indices 4-7)
         real_to_q8(0.2), real_to_q8(0.5),   // n1: w10, w11
         real_to_q8(0.3), real_to_q8(0.4),   // n0: w00, w01
         // Layer 0 weights (indices 0-3)
         real_to_q8(0.8), real_to_q8(0.1),   // n1: w10, w11
         real_to_q8(0.7), real_to_q8(0.9)    // n0: w00, w01
      };

      // Set biases (4 total: layer0 has 2, layer1 has 2)
      b_vals = {
         // Layer 1 biases
         real_to_q8(0.5),   // b[1]
         real_to_q8(1.0),   // b[0]
         // Layer 0 biases
         real_to_q8(0.6),   // b[1]
         real_to_q8(1.2)    // b[0]
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
      $display("Weight gradients (dL_dw):");
      $display("  Layer 0:");
      $write("    [");
      $write("%.4f, ", q8_to_real(dL_dw[0*16 +: 16]));
      $write("%.4f", q8_to_real(dL_dw[1*16 +: 16]));
      $display("]");
      $write("    [");
      $write("%.4f, ", q8_to_real(dL_dw[2*16 +: 16]));
      $write("%.4f", q8_to_real(dL_dw[3*16 +: 16]));
      $display("]");
      $display("  Layer 1:");
      $write("    [");
      $write("%.4f, ", q8_to_real(dL_dw[4*16 +: 16]));
      $write("%.4f", q8_to_real(dL_dw[5*16 +: 16]));
      $display("]");
      $write("    [");
      $write("%.4f, ", q8_to_real(dL_dw[6*16 +: 16]));
      $write("%.4f", q8_to_real(dL_dw[7*16 +: 16]));
      $display("]");
      $display("");

      // Display bias gradients
      $display("Bias gradients (dL_db):");
      $write("  Layer 0: [");
      $write("%.4f, ", q8_to_real(dL_db[0*16 +: 16]));
      $write("%.4f", q8_to_real(dL_db[1*16 +: 16]));
      $display("]");
      $write("  Layer 1: [");
      $write("%.4f, ", q8_to_real(dL_db[2*16 +: 16]));
      $write("%.4f", q8_to_real(dL_db[3*16 +: 16]));
      $display("]");
      $display("");

      $display("=== Test Complete ===");
      $finish;
   end

endmodule
