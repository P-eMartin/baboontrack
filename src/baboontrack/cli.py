import argparse
from .pipeline import run


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_video",
        required=True
    )

    args = parser.parse_args()

    run(
        input_video=args.input_video
    )


if __name__ == "__main__":
    main()