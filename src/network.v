module network #(
		 parameter N = 4,
		 parameter M = 4
		 ) (
		    input wire clk,
		    input wire start,
		    input wire signed [(N*16)-1:0] x,
		    input wire signed [((M*(N*N))*16)-1:0] w,
		    input wire signed [((N*M)*16)-1:0] b,
		    output wire signed [(N*16)-1:0] y,
		    output wire done
		    );
   reg [$clog2(M+1)-1:0]	count;
   wire signed [(((M+1)*N)*16)-1:0]	intermediate_states;
   wire [(M-1):0]				intermediate_done;
   wire [(M-1):0]				intermediate_start;
	      
   genvar			i;

   assign intermediate_states[(N*16)-1:0] = x;
   assign intermediate_start[0] = start;
   assign intermediate_start[M-1:1] = intermediate_done[M-2:0];
   assign y = intermediate_states[(((M+1)*N)*16)-1:(((M)*N)*16)];
   assign done = intermediate_done[M-1];
   
   generate
      for (i = 0; i < M; i = i + 1) begin
	 layer #(.N(N),
		 .M(N)
		 ) neuron_inst (
				.clk(clk),
				.start(intermediate_start[i]),
				.x(intermediate_states[
						       (((i+1)*N)*16)-1:
						       ((i*N)*16)
						       ]),
				.w(w[(((i+1)*(N*N))*16)-1:
				     ((i*(N*N))*16)
				     ]),
				.b(b[(((i+1)*N)*16)-1:
				     ((i*N)*16)
				     ]),
				.y(intermediate_states[(((i+2)*N)*16)-1:
						       (((i+1)*N)*16)
						       ]),
				.done(intermediate_done[i])
				);
      end // for (i = 0; i < M; i = i + 1)
   endgenerate

endmodule 
      

   
   
   
   
