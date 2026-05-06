import pytest

torch = pytest.importorskip("torch")

from stack.model_finetune import ICL_FinetunedModel


def test_finetuned_model_uses_mixins():
    model = ICL_FinetunedModel(
        n_genes=4,
        n_cells=3,
        n_hidden=2,
        token_dim=2,
        n_layers=1,
        n_heads=1,
        mlp_ratio=1,
        dropout=0.0,
    )

    ones = torch.ones(1, 3, 4)
    masked, mask = model.apply_finetune_mask(ones)
    assert masked.shape == ones.shape
    assert mask.shape == ones.shape

    observed = torch.rand(1, 3, 4)
    with torch.no_grad():
        output = model(
            observed,
            observed,
            mask_genes=False,
            return_loss=False,
        )

    assert "nb_mean" in output
    assert model.query_pos_embedding.shape == (model.n_hidden, model.token_dim)
