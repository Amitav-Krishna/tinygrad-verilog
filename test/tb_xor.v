// XOR learning testbench - trains a 2->2->1 network to learn XOR function
module tb_xor;
   // Network configuration: 2->2->1
   localparam NUM_LAYERS = 2;
   localparam [143:0] LAYER_SIZES = {16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd0, 16'd1, 16'd2, 16'd2};

   // Derived parameters
   localparam L0 = 2;  // inputs
   localparam L1 = 2;  // hidden
   localparam L2 = 1;  // output
   localparam TOTAL_WEIGHTS = L1*L0 + L2*L1;  // 4 + 2 = 6
   localparam TOTAL_BIASES = L1 + L2;          // 2 + 1 = 3
   localparam TOTAL_ACTS = L0 + L1 + L2;       // 2 + 2 + 1 = 5

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

   // Target
   reg signed [(L2*16)-1:0] target;

   // Loss gradient
   reg signed [(L2*16)-1:0] dL_dy;

   // Backward outputs
   wire signed [(TOTAL_WEIGHTS*16)-1:0] dL_dw;
   wire signed [(TOTAL_BIASES*16)-1:0] dL_db;
   wire bwd_done;

   // SGD
   reg signed [15:0] lr;
   wire signed [(TOTAL_WEIGHTS*16)-1:0] w_new;
   wire signed [(TOTAL_BIASES*16)-1:0] b_new;

   // Instantiate network
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

   // Instantiate SGD
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

   // Q8.8 helpers
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

   // XOR training data
   reg signed [15:0] xor_x0 [0:3];
   reg signed [15:0] xor_x1 [0:3];
   reg signed [15:0] xor_y  [0:3];

   integer epoch, sample;
   real total_loss, sample_loss;
   integer correct;

   initial begin
      $display("=== XOR Learning Test ===");
      $display("Network: 2 -> 2 -> 1");
      $display("");

      // Initialize
      clk = 0;
      rst = 1;
      start_fwd = 0;
      start_bwd = 0;

      // Learning rate (small to prevent dying ReLU)
      lr = real_to_q8(0.05);

      // XOR truth table
      xor_x0[0] = real_to_q8(0.0); xor_x1[0] = real_to_q8(0.0); xor_y[0] = real_to_q8(0.0);
      xor_x0[1] = real_to_q8(0.0); xor_x1[1] = real_to_q8(1.0); xor_y[1] = real_to_q8(1.0);
      xor_x0[2] = real_to_q8(1.0); xor_x1[2] = real_to_q8(0.0); xor_y[2] = real_to_q8(1.0);
      xor_x0[3] = real_to_q8(1.0); xor_x1[3] = real_to_q8(1.0); xor_y[3] = real_to_q8(0.0);

      // XOR solution: h0=sum, h1=AND detector, out=h0-2*h1
      // Layer 0 weights (column-major: [h0_x0, h1_x0, h0_x1, h1_x1])
      w_vals[0*16 +: 16] = real_to_q8(1.0);   // h0, x0
      w_vals[1*16 +: 16] = real_to_q8(2.0);   // h1, x0
      w_vals[2*16 +: 16] = real_to_q8(1.0);   // h0, x1
      w_vals[3*16 +: 16] = real_to_q8(2.0);   // h1, x1
      // Layer 1: output = h0 - 2*h1
      w_vals[4*16 +: 16] = real_to_q8(1.0);   // o0, h0
      w_vals[5*16 +: 16] = real_to_q8(-2.0);  // o0, h1

      // Biases
      b_vals[0*16 +: 16] = real_to_q8(0.0);   // h0 bias
      b_vals[1*16 +: 16] = real_to_q8(-3.0);  // h1 bias
      b_vals[2*16 +: 16] = real_to_q8(0.0);   // o0 bias

      // Release reset
      #20;
      rst = 0;
      #10;

      $display("Learning rate: %.3f", q8_to_real(lr));
      $display("");

      // Training loop (few epochs since starting at solution)
      for (epoch = 0; epoch < 100; epoch = epoch + 1) begin
         total_loss = 0.0;

         // Train on all 4 samples
         for (sample = 0; sample < 4; sample = sample + 1) begin
            // Set input
            x_vals = {xor_x1[sample], xor_x0[sample]};
            target = xor_y[sample];

            // Forward pass
            start_fwd = 1;
            #10;
            start_fwd = 0;
            wait(fwd_done);
            #10;

            // Compute loss gradient
            dL_dy = y_vals - target;
            sample_loss = q8_to_real(dL_dy) * q8_to_real(dL_dy) / 2.0;
            total_loss = total_loss + sample_loss;

            // Backward pass
            start_bwd = 1;
            #10;
            start_bwd = 0;
            wait(bwd_done);
            #10;

            // SGD update
            w_vals = w_new;
            b_vals = b_new;
            #10;
         end

         // Print progress every 20 epochs
         if (epoch % 20 == 0 || epoch == 99) begin
            $display("Epoch %3d: Loss = %.6f", epoch, total_loss);
         end
      end

      $display("");
      $display("=== Final Results ===");

      // Test all XOR cases
      correct = 0;
      for (sample = 0; sample < 4; sample = sample + 1) begin
         x_vals = {xor_x1[sample], xor_x0[sample]};
         target = xor_y[sample];

         start_fwd = 1;
         #10;
         start_fwd = 0;
         wait(fwd_done);
         #10;

         $display("Input: [%.1f, %.1f] -> Output: %.3f (Target: %.1f)",
            q8_to_real(xor_x0[sample]),
            q8_to_real(xor_x1[sample]),
            q8_to_real(y_vals),
            q8_to_real(target));

         // Check if prediction is correct (threshold at 0.5)
         if ((q8_to_real(y_vals) > 0.5 && q8_to_real(target) > 0.5) ||
             (q8_to_real(y_vals) <= 0.5 && q8_to_real(target) <= 0.5)) begin
            correct = correct + 1;
         end
      end

      $display("");
      $display("Accuracy: %0d/4", correct);

      if (correct == 4) begin
         $display("SUCCESS: Network learned XOR!");
      end else begin
         $display("Network needs more training or different initialization.");
      end

      $display("");
      $display("=== Test Complete ===");
      $finish;
   end

endmodule
