import torch
from lightning.pytorch.cli import LightningArgumentParser, LightningCLI

from yohane_fa.data import TimingsDataset
from yohane_fa.model import YohaneForcedAligner


class YohaneFALightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.link_arguments(
            "data",
            "model.config.init_args.vocab_size",
            compute_fn=lambda data: data.processor.tokenizer.vocab_size,
            apply_on="instantiate",
        )
        parser.link_arguments(
            "data",
            "model.config.init_args.pad_token_id",
            compute_fn=lambda data: data.processor.tokenizer.pad_token_id,
            apply_on="instantiate",
        )


def cli_main():
    torch.set_float32_matmul_precision("high")
    _ = YohaneFALightningCLI(YohaneForcedAligner, TimingsDataset)


if __name__ == "__main__":
    cli_main()
