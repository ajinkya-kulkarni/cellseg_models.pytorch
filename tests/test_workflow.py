import numpy as np
from PIL import Image
from tifffile import imwrite

from stardist_minimal.evaluate import main as evaluate_main
from stardist_minimal.predict import main as predict_main
from stardist_minimal.train import main as train_main


def _make_split(root, name):
    images = root / name / "images"
    masks = root / name / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for i in range(2):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        image[8:24, 8:24] = (180, 90, 40)
        mask = np.zeros((32, 32), dtype=np.int32)
        mask[10:22, 10:22] = 1
        Image.fromarray(image).save(images / f"{i}.png")
        imwrite(masks / f"{i}.tif", mask)
    return images, masks


def test_train_evaluate_predict(tmp_path):
    train_images, train_masks = _make_split(tmp_path, "train")
    val_images, val_masks = _make_split(tmp_path, "val")
    output = tmp_path / "run"

    train_main(
        [
            "--train-images",
            str(train_images),
            "--train-masks",
            str(train_masks),
            "--val-images",
            str(val_images),
            "--val-masks",
            str(val_masks),
            "--output-dir",
            str(output),
            "--encoder",
            "resnet18",
            "--n-rays",
            "8",
            "--patch-size",
            "32",
            "--batch-size",
            "2",
            "--epochs",
            "1",
            "--num-workers",
            "0",
            "--no-pretrained",
            "--device",
            "cpu",
        ]
    )

    checkpoint = output / "best.pt"
    assert checkpoint.exists()
    assert (output / "last.pt").exists()
    assert (output / "history.json").exists()

    predictions = tmp_path / "eval_predictions"
    evaluate_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--images",
            str(val_images),
            "--masks",
            str(val_masks),
            "--device",
            "cpu",
            "--score-thresh",
            "0.99",
            "--save-dir",
            str(predictions),
        ]
    )
    assert (predictions / "0.npy").exists()

    inference = tmp_path / "predictions"
    predict_main(
        [
            "--checkpoint",
            str(checkpoint),
            "--input",
            str(val_images / "0.png"),
            "--output-dir",
            str(inference),
            "--device",
            "cpu",
            "--score-thresh",
            "0.99",
        ]
    )
    labels = np.load(inference / "0.npy")
    assert labels.shape == (32, 32)
    assert labels.dtype == np.int32
