// SGD optimizer module - TODO: implement
module sgd #(
	     parameter N = 4,
	     parameter M = 4
	     ) (
		input wire signed [((M*(N*N))*16)-1:0]	w,
		input wire signed [((N*M)*16)-1:0]	b,
		input wire signed [((M*(N*N))*16)-1:0]	dL_dw,
		input wire signed [((N*M)*16)-1:0]	dL_db,
		input wire signed [15:0] lr,
		output wire signed [((M*(N*N))*16)-1:0]	w_new,
		output wire signed [((N*M)*16)-1:0]	b_new
		);
   wire [(((M*(N*N))*16)+((N*M)*16))-1:0] params;
   wire [(((M*(N*N))*16)+((N*M)*16))-1:0] gradients;
   wire [(((M*(N*N))*16)+((N*M)*16))-1:0] params_new;
   assign params = {w, b};
   assign gradients = {dL_dw, dL_db};
   assign w_new = params_new[((N*M)*16) +: ((M*(N*N))*16)];
   assign b_new = params_new[((N*M)*16)-1:0];
   
		  
   genvar i;

   generate
      for (i = 0; i < (((M*(N*N)))+((N*M))); i = i + 1) begin : param_update
         wire signed [31:0] scaled_grad;
         assign scaled_grad = lr * gradients[i*16 +: 16];  // Q8.8 * Q8.8 = Q16.16
         assign params_new[i*16 +: 16] = params[i*16 +: 16] - scaled_grad[23:8];
      end
   endgenerate
   

endmodule
