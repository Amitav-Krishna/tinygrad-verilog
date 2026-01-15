module layer #(
		parameter N_IN  = 4,  // number of inputs
		parameter N_OUT = 4   // number of neurons (outputs)
	       ) (
		   input wire		      clk,
		   input wire		      start,
		   input wire signed [(N_IN*16)-1:0]      x,
		   input wire signed [((N_OUT*N_IN)*16)-1:0] w,
		   input wire signed [(N_OUT*16)-1:0]  b,
		   output wire signed [(N_OUT*16)-1:0] y,
		   output wire done
		  );
   reg [$clog2(N_IN+1)-1:0] count;
   wire signed [(N_OUT*16)-1:0] activations;
   reg signed [15:0]	    x_val;
   reg signed [(N_OUT*16)-1:0] w_vals;
   reg start_delayed;
   
   genvar	    i;

   assign y = activations;

   wire [N_OUT-1:0] neurons_done;

   generate
      for (i = 0; i < N_OUT; i = i + 1) begin : neurons
	 neuron #(.N(N_IN)) neuron_inst (
				      .clk(clk),
				      .start(start_delayed),
				      .x(x_val),
		      .w(w_vals[
				((i+1)*16)-1:
				(i*16)		
				]),
		      .b(b[((i+1)*16)-1:(i*16)]),
	      .activation(activations[((i+1)*16)-1:(i*16)]),
	      .done(neurons_done[i])
	      );
      end // for (i = 0; i < M; i++)
   endgenerate
   assign done = &neurons_done;
   
   always @(posedge clk) begin
      start_delayed <= start;
      
      if (start) begin
 	 count <= 0;
 	 x_val <= x[0 +: 16];
 	 w_vals <= w[0 +: (N_OUT*16)];
      end else if (count < N_IN - 1) begin
 	 count <= count + 1;
 	 x_val <= x[(((count + 1) * 16)) +: 16];
 	 w_vals <= w[(((count + 1) * N_OUT * 16)) +: (N_OUT*16)];
      end
   end
	 
endmodule // layer

