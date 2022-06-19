import os

import numpy
import PIL
import PIL.Image

from tqdm import tqdm

import torch

from run import estimate


SOURCE: str = "C:\\Users\\fcocl\\Datasets\\sketchy\\photo\\tx_000000000000"
TARGET: str = "C:\\Users\\fcocl\\Datasets\\sketchy\\photo\\tx_000000000000_edges"


def convert_and_save_image(input_path: str, output_path: str) -> None:
    tenInput = torch.FloatTensor(
        numpy.ascontiguousarray(
            numpy.array(PIL.Image.open(input_path))[:, :, ::-1]
            .transpose(2, 0, 1)
            .astype(numpy.float32)
            * (1.0 / 255.0)
        )
    )
    tenOutput = estimate(tenInput)
    PIL.Image.fromarray(
        (
            torch.clamp(tenOutput, 0.0, 1.0).numpy().transpose(1, 2, 0)[:, :, 0] * 255.0
        ).astype(numpy.uint8)
    ).save(output_path)


if __name__ == "__main__":
    total = len(next(os.walk(SOURCE))[1])
    for i, directory in enumerate(next(os.walk(SOURCE))[1]):
        for image in tqdm(
            next(os.walk(os.path.join(SOURCE, directory)))[2],
            desc=f"Converting the {directory} class. {i}/{total}",
        ):
            convert_and_save_image(
                os.path.join(SOURCE, directory, image),
                os.path.join(TARGET, directory, image),
            )
