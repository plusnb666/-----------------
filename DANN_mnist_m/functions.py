from torch.autograd import Function


"""DANN 的梯度反转层（Gradient Reversal Layer, GRL）。

前向传播时 GRL 不改变特征值；反向传播时将梯度乘以 ``-alpha``。
因此特征提取器会在保持任务分类能力的同时，主动学习难以区分源域和
目标域的领域不变表示。
"""


class ReverseLayerF(Function):
    """实现领域对抗训练所需的梯度反转算子。"""

    @staticmethod
    def forward(ctx, x, alpha):
        """前向直接返回输入，并保存梯度反转强度。"""
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """反向传播时反转并缩放来自领域分类器的梯度。"""
        output = grad_output.neg() * ctx.alpha

        return output, None

