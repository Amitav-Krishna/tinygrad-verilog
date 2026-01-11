module layer #(
		parameter N = 4,
		parameter M = 4
	       ) (
		   input wire			      clk,
		   input wire			      start,
		   input wire signed [(N*16)-1:0]     x,
		   input wire signed [((M*N)*16)-1:0] w,
		   input wire signed [(M*16)-1:0]  b,
		   output wire signed [(M*16)-1:0] y,
		   output wire done
		  );
   
   reg [$clog2(N+1)-1:0] count; 
   wire signed [(M*16)-1:0] activations;
   wire signed [15:0]	    x_val;
   wire signed [(M*16)-1:0] w_vals;
   
   genvar		    i;

   generate
      for (i = 0; i < M; i++) begin
	 neuron #(.N(N)) neuron_inst (
				      .clk(clk),
				      .start(start),
				      .x(x_val),
				      .w(w_vals[
						((i+1)*16)-1:
						(i*16)			
						]),
				      .b(b[(16*(i+1))-1:16*i]),
				      .activation(activation[(16*i)-1:16*i]),
				      .done(done)
				      );
      end // for (i = 0; i < M; i++)
   endgenerate

   always @(posedge clk) begin
      if (start) begin
	 count <= 1;
	 x_val <= x[15:0];
	 w_vals <= w[(M*16)-1:0];
	 done <= (N == 1);
      end else if (count > 0 && count < N) begin
	 x_val <= x[((count+1)*1*16)-1:(count*1*16)];
	 w_vals <= w[((count+1)*M*16)-1:(count*M*16)];
	 count <= count + 1;
	 done <= (count == N - 1);
      end else if (done) begin
	 done <= 0;
      end
   end
	 
endmodule // layer

