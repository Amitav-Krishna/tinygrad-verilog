module tb_mse;

   reg signed [15:0] y_pred;
   reg signed [15:0] y_true;

   wire signed [15:0] loss;
   
   mse uut (
	    .y_pred(y_pred),
	    .y_true(y_true),
	    .loss(loss)
	    );

   function real q8_to_real;
      input signed [15:0] q;
      begin
         q8_to_real = $itor(q) / 256.0;
      end
   endfunction

   // Helper: convert real to Q8.8
   function signed [15:0] real_to_q8;
      input real r;
      begin
         real_to_q8 = $rtoi(r * 256.0);
      end
   endfunction // real_to_q8

   initial begin
      $display("=== MSE Unit Test");


      y_pred = real_to_q8(4.0);
      y_true = real_to_q8(2.0);

      #10;

      $display("MSE(%.3f, %.3f) = %.3f",
	       q8_to_real(y_pred), q8_to_real(y_true), q8_to_real(loss));
      $finish;
   end // initial begin
endmodule // tb_mse

	
