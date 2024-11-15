import torch
from . import torch_ops_matmul

class MatmulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = torch.zeros_like(a)
        # 计算前向传播结果
        torch_ops_matmul.forward(c, a, b)
        # 保存张量以供反向传播使用
        ctx.save_for_backward(a, b, c)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # 从ctx中恢复保存的张量
        a, b, c = ctx.saved_tensors
        # 根据计算过程定义各个输入的梯度，device需要和输入相同
        # 如果某个输入不需要梯度，返回None
        grad_a = grad_output * b
        grad_b = grad_output * a
        return grad_a, grad_b

matmul_op = MatmulOp.apply