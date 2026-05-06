import pytest


torch = pytest.importorskip("torch")

from stack.model import StateICLModel, scShiftAttentionModel


@pytest.fixture()
def small_stack_model():
    return StateICLModel(
        n_genes=6,
        n_cells=4,
        n_hidden=2,
        token_dim=8,
        n_layers=1,
        n_heads=2,
        mlp_ratio=1,
        dropout=0.0,
    )


def test_stackmodel_forward_shapes(small_stack_model):
    model = small_stack_model
    model.train()

    features = torch.rand(2, 4, 6)
    output = model(features, return_loss=False)

    assert set(output) >= {"nb_mean", "nb_dispersion", "px_scale"}
    assert output["nb_mean"].shape == (2, 4, 6)
    assert output["nb_dispersion"].shape == (2, 4, 6)
    assert output["px_scale"].shape == (2, 4, 6)


def test_inference_mixin_predict_uses_forward(small_stack_model):
    model = small_stack_model
    model.eval()

    features = torch.rand(1, 4, 6)
    with torch.no_grad():
        predictions = model.predict(features)

    assert set(predictions) >= {"nb_mean", "nb_dispersion", "px_scale"}
    assert predictions["nb_mean"].shape == (1, 4, 6)


def test_scshiftattentionmodel_aliases_stackmodel():
    assert scShiftAttentionModel is StateICLModel
