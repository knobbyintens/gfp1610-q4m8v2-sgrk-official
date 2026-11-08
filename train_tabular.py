"""Training entry point for GRAK (preprint 2601.09999)."""

from grak.model import GatedRecurrentAttentionKernel


def main():
    model = GatedRecurrentAttentionKernel(features=32)
    print("training GRAK on tabular forecasting data")


if __name__ == "__main__":
    main()
