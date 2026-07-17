import torch
from torch import nn

from mixedconv.layers import apply_precision_map, build_linear_registry, identity_precision_map


class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.to_q = nn.Linear(8, 8)
        self.w1 = nn.Linear(8, 16)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.to_q(inputs) + self.w1(inputs)[..., :8]


def test_identity_swap_is_bit_exact() -> None:
    torch.manual_seed(7)
    model = nn.ModuleDict({"transformer_blocks": nn.ModuleList([TinyBlock()])})
    block = model["transformer_blocks"][0]
    inputs = torch.randn(2, 4, 8)
    before = block(inputs)

    records = build_linear_registry(model)
    apply_precision_map(model, identity_precision_map(records))
    after = block(inputs)

    assert torch.equal(before, after)
    assert [record.functional_type for record in records] == ["attention_q", "swiglu_w1"]
