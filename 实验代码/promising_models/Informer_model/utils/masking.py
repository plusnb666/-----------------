import torch


"""Informer 注意力掩码。

掩码使用布尔张量表示，``True`` 的位置会在 softmax 前被填充为负无穷，
从而禁止模型读取未来时间步。
"""


class TriangularCausalMask():
    """标准上三角因果掩码，禁止每个位置访问未来位置。"""

    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask

class ProbMask():
    """只为 ProbSparse 选中的查询生成因果掩码。"""

    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                             torch.arange(H)[None, :, None],
                             index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)
    
    @property
    def mask(self):
        return self._mask