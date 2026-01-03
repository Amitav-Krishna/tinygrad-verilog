# Verilog Neural Network Makefile

# Icarus Verilog compiler and simulator
IVERILOG = iverilog
VVP = vvp

# Directories
SRC = src
TEST = test
BUILD = build

# Create build directory
$(BUILD):
	mkdir -p $(BUILD)

# === Individual module tests ===

# Test adder (hello world)
test_adder: $(BUILD)
	$(IVERILOG) -o $(BUILD)/adder.vvp $(SRC)/adder.v $(TEST)/tb_adder.v
	$(VVP) $(BUILD)/adder.vvp

# Test fixed-point multiply
test_mul: $(BUILD)
	$(IVERILOG) -o $(BUILD)/mul.vvp $(SRC)/fixed_mul.v $(TEST)/tb_fixed_mul.v
	$(VVP) $(BUILD)/mul.vvp

# Test MAC
test_mac: $(BUILD)
	$(IVERILOG) -o $(BUILD)/mac.vvp $(SRC)/fixed_mul.v $(SRC)/mac.v $(TEST)/tb_mac.v
	$(VVP) $(BUILD)/mac.vvp

# Test ReLU
test_relu: $(BUILD)
	$(IVERILOG) -o $(BUILD)/relu.vvp $(SRC)/relu.v $(TEST)/tb_relu.v
	$(VVP) $(BUILD)/relu.vvp

# Test activation
test_activation: $(BUILD)
	$(IVERILOG) -o $(BUILD)/activation.vvp $(SRC)/fixed_mul.v $(SRC)/mac.v $(SRC)/relu.v $(SRC)/activation.v $(TEST)/tb_activation.v
	$(VVP) $(BUILD)/activation.vvp
# Test neuron
test_neuron: $(BUILD)
	$(IVERILOG) -o $(BUILD)/neuron.vvp $(SRC)/activation.v $(SRC)/fixed_mul.v $(SRC)/mac.v $(SRC)/relu.v $(SRC)/neuron.v $(TEST)/tb_neuron.v
	$(VVP) $(BUILD)/neuron.vvp

# Test full network
test_network: $(BUILD)
	$(IVERILOG) -o $(BUILD)/network.vvp $(SRC)/*.v $(TEST)/tb_network.v
	$(VVP) $(BUILD)/network.vvp

# Clean build artifacts
clean:
	rm -rf $(BUILD)

.PHONY: test_adder test_mul test_mac test_relu test_neuron test_network clean
