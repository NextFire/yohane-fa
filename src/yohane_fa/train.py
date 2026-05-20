import torch
from lightning.pytorch.cli import LightningCLI

from yohane_fa.data import TimingsDataset
from yohane_fa.model import YohaneForcedAligner


class YohaneFALightningCLI(LightningCLI):
    pass


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = YohaneFALightningCLI(YohaneForcedAligner, TimingsDataset)


if __name__ == "__main__":
    cli_main()
