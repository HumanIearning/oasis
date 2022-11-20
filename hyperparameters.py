import argparse

def define_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        '--freeze_encoder',
        action='store_true'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=0.01
    )
    p.add_argument(
        '--n_epoch',
        type=int,
        default=4
    )
    p.add_argument(
        '--batch_size',
        type=int,
        default=16
    )

    return p.parse_args()