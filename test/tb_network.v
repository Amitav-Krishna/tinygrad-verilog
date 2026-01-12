module tb_network;
   reg clk;

   reg start;

   reg signed [31:0] x_vals;
   reg signed [127:0] w_vals;
   reg signed [63:0]  b_vals;

   wire signed [31:0] y_vals;
   wire		      done;


   network #(
	     .N(2),
	     .M(2)
	     ) uut (
		    .clk(clk),
		    .start(start),
		    .x(x_vals),
		    .w(w_vals),
		    .b(b_vals),
		    .y(y_vals),
		    .done(done)
		    );
   always #5 clk = ~clk;

   // ~ is the not operator, which makes clock flip every 5 time units.
   
   // Helper functions
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

   initial begin
      $display("=== Network Unit Test ===");

      x_vals = {
		real_to_q8(2.0),
		real_to_q8(1.0)
		};

      w_vals = {
		real_to_q8(0.2), real_to_q8(0.5), real_to_q8(0.3), real_to_q8(0.4),

		real_to_q8(0.8), real_to_q8(0.1), real_to_q8(0.7), real_to_q8(0.9)
		};

      b_vals = {
		real_to_q8(1), real_to_q8(0.5),

		real_to_q8(1.2), real_to_q8(0.6)
		};

      clk = 0;
      start = 0;


      #20;
      start = 1;
      #10;
      start = 0;

      #400;

      $display("Inputs:  [%.3f, %.3f]",
	       q8_to_real(x_vals[15:0]),
	       q8_to_real(x_vals[31:16])
	       );
      

      $display("Outputs: [%.3f, %.3f]",
	       q8_to_real(y_vals[15:0]),
	       q8_to_real(y_vals[31:16])
	       );
      $display("Done: %b", done);
   end // initial begin
endmodule // tb_network
