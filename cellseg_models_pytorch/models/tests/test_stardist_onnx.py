import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch

from cellseg_models_pytorch.models.stardist import (
    StarDistONNXWrapper,
    export_stardist_onnx,
)
from cellseg_models_pytorch.models.stardist.stardist_unet import stardist_nuclei


def _make_stardist() -> torch.nn.Module:
    return stardist_nuclei(
        n_rays=8,
        n_nuc_classes=3,
        enc_name="resnet18",
        enc_pretrain=False,
    ).eval()


def test_stardist_onnx_wrapper_matches_model_output() -> None:
    model = _make_stardist()
    wrapper = StarDistONNXWrapper(model).eval()
    x = torch.rand(1, 3, 64, 64)

    with torch.inference_mode():
        expected = model(x)["nuc"]
        binary_map, ray_map, type_map = wrapper(x)

    assert expected.binary_map is not None
    torch.testing.assert_close(binary_map, expected.binary_map)
    torch.testing.assert_close(ray_map, expected.aux_map)
    torch.testing.assert_close(type_map, expected.type_map)


def test_export_stardist_onnx_validates_input_shape(tmp_path: Path) -> None:
    model = _make_stardist()

    with pytest.raises(ValueError, match="four positive BCHW dimensions"):
        export_stardist_onnx(
            model,
            tmp_path / "stardist.onnx",
            input_shape=(1, 3, 64, 0),
        )


def test_export_stardist_onnx_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = _make_stardist()
    model.train()
    output_path = tmp_path / "nested" / "stardist.onnx"
    export_call = {}

    monkeypatch.setitem(sys.modules, "onnx", ModuleType("onnx"))

    def fake_export(*args, **kwargs) -> None:
        export_call["args"] = args
        export_call["kwargs"] = kwargs

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    result = export_stardist_onnx(
        model,
        output_path,
        input_shape=(1, 3, 64, 64),
        dynamic_batch=True,
    )

    assert result == output_path
    assert output_path.parent.is_dir()
    assert model.training
    assert export_call["kwargs"]["input_names"] == ["image"]
    assert export_call["kwargs"]["output_names"] == [
        "binary_map",
        "ray_map",
        "type_map",
    ]
    assert export_call["kwargs"]["dynamic_axes"] == {
        "image": {0: "batch"},
        "binary_map": {0: "batch"},
        "ray_map": {0: "batch"},
        "type_map": {0: "batch"},
    }
    assert export_call["kwargs"]["opset_version"] == 17


def test_stardist_onnxruntime_matches_pytorch(tmp_path: Path) -> None:
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    model = _make_stardist()
    wrapper = StarDistONNXWrapper(model).eval()
    output_path = export_stardist_onnx(
        model,
        tmp_path / "stardist.onnx",
        input_shape=(1, 3, 64, 64),
        dynamic_batch=True,
    )

    session = ort.InferenceSession(
        str(output_path), providers=["CPUExecutionProvider"]
    )
    x = torch.rand(2, 3, 64, 64)

    with torch.inference_mode():
        expected = [tensor.cpu().numpy() for tensor in wrapper(x)]

    actual = session.run(None, {"image": x.numpy()})

    assert [output.name for output in session.get_outputs()] == [
        "binary_map",
        "ray_map",
        "type_map",
    ]
    for expected_tensor, actual_tensor in zip(expected, actual):
        np.testing.assert_allclose(
            actual_tensor,
            expected_tensor,
            rtol=1e-4,
            atol=1e-5,
        )
