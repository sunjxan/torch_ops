import torch
import torch_ops_matmul
# torch.ops.load_library('torch_ops_matmul.so')
# torch_ops_matmul = torch.ops.torch_ops_matmul
# from torch.utils.cpp_extension import load
# torch_ops_matmul = load("torch_ops_matmul", sources=["src/cuda/matmul.cu"], verbose=True)

class MatmulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # 计算前向传播结果，要求确保输入tensor元素地址连续
        c = torch_ops_matmul.forward(a.contiguous(), b.contiguous())
        # 保存张量以供反向传播使用
        ctx.save_for_backward(a, b, c)
        # 如果算子不需要考虑反向传播，可以将函数的输出标记不需要微分
        # ctx.mark_non_differentiable(c)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # 从ctx中恢复保存的张量
        a, b, c = ctx.saved_tensors
        # 根据外层梯度grad_output和forward过程计算各个输入的梯度，device需要和输入相同
        # 如果某个输入不需要梯度，返回None
        g = grad_output.contiguous()
        a_T = a.T.contiguous()
        b_T = b.T.contiguous()
        grad_a = torch_ops_matmul.forward(g, b_T)
        grad_b = torch_ops_matmul.forward(a_T, g)
        return grad_a, grad_b

matmul = MatmulOp.apply
