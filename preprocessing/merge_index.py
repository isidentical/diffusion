import fire
from streaming.base.util import merge_index


def main(out_dir_root: str = "train_mds"):
    merge_index(out_dir_root, keep_local=True)


if __name__ == "__main__":
    fire.Fire(main)
