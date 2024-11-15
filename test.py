import torch, op
a = torch.ones(5, requires_grad=True, device="cuda:0")
b = torch.ones(5, requires_grad=True, device="cuda:0")
c = op.matmul_op(a, b)
c.sum().backward()
print(a.grad, b.grad, c.grad)