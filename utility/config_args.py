from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter, ArgumentTypeError


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def get_arguments():
    # Parse command line arguments
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument("-aw", "--ANOMALY_WEIGHT", default=0.7, type=float, help="anomaly weight in range 0-9 (default: 0.9)")
    parser.add_argument("-lr", "--LEARNING_RATE", default=0.002, type=float, help="learning rate (default: 0.002)")
    parser.add_argument("-metastep", "--META_STEP_SIZE", default=0.25, type=float, help="meta step size (default: 0.25)")
    parser.add_argument("-bs", "--BATCH_SIZE", default=25, type=int, help="batch size (default: 25)")
    parser.add_argument("-metaiter", "--META_ITERS", default=1500, type=int, help="Meta iterations / outer (default: 1500)")
    parser.add_argument("-ii", "--INNER_ITERS", default=4, type=int, help="Inner iterations / inner (default: 4)")
    parser.add_argument("-ei", "--EVAL_INTERVAL", default=100, type=int, help="Evaluation interval per number of iteration default: 100")
    
    parser.add_argument("-dn", "--DATASET_NAME", default="mura", help="name of dataset in data directory.")
    parser.add_argument("-sz", "--SIZE", default=128, type=int, help="size of cropped image.")
    parser.add_argument("-ss", "--STEP_SLIDING", default=20, type=int, help="step size")
    parser.add_argument("-ws", "--WIN_SIZE", default=128, type=int, help="step size")
    parser.add_argument("-s", "--SHOTS", default=20, type=int, help="how many data that you want to use.")
    parser.add_argument("-nd", "--NO_DATASET", default=0, type=int, help="select which number of dataset.")
    parser.add_argument("-m", "--MODE", default=True, type=str2bool, help="Mode. Train (True) or Only Test (False)")
    parser.add_argument("-rtd", "--ROOT_DATA_DIR", default="data", help="Root directory of data")
    parser.add_argument("-ted", "--TEST_DATA", default="test_data", help="Directory of test data")
    parser.add_argument("-trd", "--TRAIN_DATA", default="train_data", help="Directory of train data")
    parser.add_argument("-eld", "--EVAL_DATA", default="eval_data", help="Directory of evaluation data")
    parser.add_argument("-rd", "--RESULT_DIR", default="result/", help="Directory of result")
    parser.add_argument("-smd", "--SAVED_MODEL_DIR", default="saved_model", help="Directory of saved_model")
    parser.add_argument("-bmw", "--BEST_MODEL_WEIGHT", default=True, type=str2bool, help="Using the best model weight")
    parser.add_argument("-ks", "--KERNEL_SIZE", default=4, type=int, help="kernel size in generator decoder.")
    
    return parser
