"""Simple tinygrad example demonstrating basic operations and a tiny neural network."""
from tinygrad import Tensor
from tinygrad import nn

# Basic tensor operations
print("=== Basic Tensor Operations ===")
a = Tensor([1, 2, 3, 4])
b = Tensor([5, 6, 7, 8])
print(f"a = {a.numpy()}")
print(f"b = {b.numpy()}")
print(f"a + b = {(a + b).numpy()}")
print(f"a * b = {(a * b).numpy()}")
print(f"a.sum() = {a.sum().numpy()}")

# Matrix operations
print("\n=== Matrix Operations ===")
x = Tensor([[1, 2], [3, 4]])
w = Tensor([[5, 6], [7, 8]])
print(f"x @ w = \n{(x @ w).numpy()}")

# Simple neural network forward pass
print("\n=== Simple Neural Network ===")
class TinyNet:
    def __init__(self):
        self.l1 = nn.Linear(2, 4)   # 2 inputs -> 4 hidden
        self.l2 = nn.Linear(4, 1)   # 4 hidden -> 1 output

    def __call__(self, x: Tensor) -> Tensor:
        x = self.l1(x).relu()
        return self.l2(x)

model = TinyNet()
inp = Tensor([[0.5, 0.5]])
out = model(inp)
print(f"Input: {inp.numpy()}")
print(f"Output: {out.numpy()}")

# Training loop example (XOR problem)
print("\n=== Training XOR ===")
X = Tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = Tensor([[0], [1], [1], [0]])

model = TinyNet()
opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.1)

Tensor.training = True
for step in range(200):
    opt.zero_grad()
    pred = model(X)
    loss = ((pred - Y) ** 2).mean()  # MSE loss
    loss.backward()
    opt.step()
    if step % 50 == 0:
        print(f"Step {step}: loss = {loss.numpy():.4f}")

Tensor.training = False
print("\nFinal predictions:")
print(f"Input: {X.numpy().tolist()}")
print(f"Target: {Y.numpy().flatten().tolist()}")
print(f"Predicted: {[round(p, 2) for p in model(X).numpy().flatten().tolist()]}")
