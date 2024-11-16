import torch
from . import torch_ops_matmul
# torch.ops.load_library('op/torch_ops_matmul.so')
# torch_ops_matmul = torch.ops.torch_ops_matmul
# from torch.utils.cpp_extension import load
# torch_ops_matmul = load("torch_ops_matmul", sources=["src/matmul_kernel.cu", "src/matmul.cpp"], extra_include_paths=["include"], verbose=True)

class MatmulOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        # 计算前向传播结果
        c = torch_ops_matmul.forward(a, b)
        # 保存张量以供反向传播使用
        ctx.save_for_backward(a, b, c)
        return c

    @staticmethod
    def backward(ctx, grad_output):
        # 从ctx中恢复保存的张量
        a, b, c = ctx.saved_tensors
        # 根据外层梯度grad_output和forward过程计算各个输入的梯度，device需要和输入相同
        # 如果某个输入不需要梯度，返回None
        grad_a = torch_ops_matmul.backward_left(a, b)
        grad_b = torch_ops_matmul.backward_right(a, b)
        return grad_a, grad_b

matmul_op = MatmulOp.apply
