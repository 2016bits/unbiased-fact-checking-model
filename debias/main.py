import argparse

from utils.log import get_logger

def main(args):
    # init logger
    log_path = args.log_path + "debias.log"
    logger = get_logger
    # load data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_path", type=str, default='./logs/')

    args = parser.parse_args()
    main(args)
