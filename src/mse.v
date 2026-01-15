module mse (
	    input wire [15:0] y_pred,
	    input wire [15:0] y_true,
	    output wire [15:0] loss
	    );
   wire signed [15:0] error;
   wire signed [31:0] error_sq;
   assign error = (y_pred - y_true);
   assign error_sq = error * error;
   assign loss = error_sq[23:8];

endmodule
   
   
