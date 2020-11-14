import argparse


def parse_arguments():
    # Command-line flags are defined here.
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        default=10, help="Number of epochs to train on.")
    parser.add_argument('--lr', dest='lr', type=float,
                        default=3e-4, help="The learning rate.")
    parser.add_argument('--model_path', dest='model_path', type=str,
                        default="", help="Optionally load an existing model")
    parser.add_argument('--train', dest='train', action='store_true',
                        help="Whether to train or run demo")
    parser.add_argument('--print_to_file', dest='print_to_file',
                        action='store_true', help="Whether to save printout to file")
    parser.set_defaults(train=False)
    parser.set_defaults(print_to_file=False)
    return parser.parse_args()
