import sys


def main():
    print("\nRunning child script...\n")

    import torch

    # Basic PyTorch verification
    print("\nPyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # Simple tensor operation
    x = torch.rand(2, 3)
    print("\nRandom tensor:\n", x)

    return 0

if __name__ == "__main__":
    print("Child script started.")
    sys.exit(main())
