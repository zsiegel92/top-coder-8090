import sys
from models_and_constants import Input
from model import predict


def parse_args():
    args = sys.argv[1:]
    if len(args) != 3:
        raise ValueError("Expected 3 arguments")
    return Input(
        trip_duration_days=int(args[0]),
        miles_traveled=float(args[1]),
        total_receipts_amount=float(args[2]),
    )


def main():
    args = parse_args()
    prediction = predict(args)
    print(prediction)


if __name__ == "__main__":
    main()
