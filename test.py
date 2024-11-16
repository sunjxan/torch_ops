import torch, op
a = torch.randn((2,3), requires_grad=True, device="cuda:0")
b = torch.randn((3,4), requires_grad=True, device="cuda:0")
c = op.matmul_op(a, b)
c.sum().backward()
d = torch.matmul(a, b)
print(torch.allclose(c, d, atol=1e-5))
print(a.grad, b.grad)
