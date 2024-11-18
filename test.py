import torch, op
a1 = torch.randn((2, 3), dtype=torch.float32, requires_grad=True, device="cuda:0")
b1 = torch.randn((3, 4), dtype=torch.float32, requires_grad=True, device="cuda:0")
c1 = torch.mm(a1, b1)
d1 = c1 * 5
d1.sum().backward()
a2 = a1.clone().detach().requires_grad_(True)
b2 = b1.clone().detach().requires_grad_(True)
c2 = op.matmul_op(a2, b2)
d2 = c2 * 5
d2.sum().backward()
print(torch.allclose(c1, c2, atol=5e-3))
print(torch.allclose(a1.grad, a2.grad, atol=5e-3))
print(torch.allclose(b1.grad, b2.grad, atol=5e-3))
