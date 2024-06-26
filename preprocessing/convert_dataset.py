import json
import re
from glob import iglob
from pathlib import Path
from multiprocessing.pool import Pool

import fire
import webdataset as wds
from braceexpand import braceexpand
from streaming import MDSWriter


def process_sample(sample):
    return {
        "image": sample["jpg"],
        "caption": json.loads(sample["json"]).get("caption", ""),
    }


def process_shard(job: tuple[Path, Path]):
    shard_path, out_dir_root = job

    dataset = wds.DataPipeline(
        wds.SimpleShardList([str(shard_path)]),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        # wds.select(filter_cond),
        wds.map(process_sample),
    )

    COLUMNS = {"image": "bytes", "caption": "str"}
    out_dir = str(out_dir_root / shard_path.name)

    with MDSWriter(out=out_dir, columns=COLUMNS, size_limit="128mb") as out:
        for sample in dataset:
            out.write(sample)

    print(f"Saved {shard_path.name}")


def main(
    wds_path: str = "data/00000.tar",
    out_dir_root: str | Path = "train_mds",
    exclude_regex: str = "",
    num_processes: int | None = None,
):
    print(wds_path, out_dir_root, exclude_regex)
    out_dir_root = Path(out_dir_root)
    shard_paths = list(map(Path, iglob(wds_path) if "*" in wds_path else braceexpand(wds_path)))
    if exclude_regex:
        shard_paths = list(filter(lambda s: not re.search(exclude_regex, str(s)), shard_paths))
    print(f"{len(shard_paths)=}")

    with Pool(num_processes) as pool:
        pool.map(process_shard, ((p, out_dir_root) for p in shard_paths))


if __name__ == "__main__":
    fire.Fire(main)
