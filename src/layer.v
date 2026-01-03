module layer #(
  parameter N = 4,
  parameter M = 4,
) (
  input wire clk,
  input wire start,
  input wire signed [(N*16)-1:0] x,
  input wire signed [((M*N)*16)-1:0] w,
  input wire signed [(M*16)-1:0] b,
  output wire signed [(M*16)-1:0] y,
  output wire done
);
  wire signed [(M*16)-1:0] activations;
  integer i;
  
  for (i = 0; i < M; i++) begin
	  neuron #(.N(N)) neuron_inst_i (
		  .clk(clk),
		  .start(start),
		  .x(x[((i+1)*16)-1:(i*16)]),
		  .w(w[((16*N)*(i+1))-1:(16*N*i)]),
		  .b(b[(16*(i+1))-1:16*i]),
		  .activation(activation[(16*i)-1:16*i]),
		  .done(done)
	  );
  end
