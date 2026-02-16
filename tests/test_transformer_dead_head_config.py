import json

from fla.models.dead_head_mask import apply_dead_head_masks_from_json
from fla.models.transformer.configuration_transformer import TransformerConfig
from fla.models.transformer.modeling_transformer import TransformerModel


def _tiny_config(**kwargs) -> TransformerConfig:
    defaults = dict(
        hidden_size=8,
        num_hidden_layers=2,
        num_heads=2,
        num_kv_heads=2,
        max_position_embeddings=16,
        vocab_size=64,
        fuse_norm=False,
        fuse_swiglu=False,
        fuse_cross_entropy=False,
        attn_impl="naive_attn",
    )
    defaults.update(kwargs)
    return TransformerConfig(**defaults)


def test_apply_dead_head_masks_to_model(tmp_path) -> None:
    mask_path = tmp_path / "dead.json"
    mask_path.write_text(
        json.dumps({"dead_heads_by_step": {"10000": [[0, 1], [1, 0]]}}),
        encoding="utf-8",
    )
    config = _tiny_config()
    model = TransformerModel(config)
    total_masked = apply_dead_head_masks_from_json(
        model=model,
        mask_path=mask_path,
        step=10000,
    )
    assert total_masked == 2
    assert model.layers[0].attn.dead_head_mask.tolist() == [False, True]
    assert model.layers[1].attn.dead_head_mask.tolist() == [True, False]
