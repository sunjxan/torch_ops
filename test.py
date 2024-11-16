import torch, op
a1 = torch.randn((2, 3), requires_grad=True, device="cuda:0")
b1 = torch.randn((3, 4), requires_grad=True, device="cuda:0")
c1 = torch.matmul(a1, b1)
d1 = c1 * 5
d1.sum().backward()
a2 = torch.tensor(a1, requires_grad=True, device="cuda:0")
b2 = torch.tensor(b1, requires_grad=True, device="cuda:0")
c2 = op.matmul_op(a2, b2)
d2 = c2 * 5
d2.sum().backward()
print(torch.allclose(c1, c2, atol=1e-5))
print(torch.allclose(a1.grad, a2.grad, atol=1e-5))
print(torch.allclose(b1.grad, b2.grad, atol=1e-5))
