import os

from data_pipeline import StreamingRecordDataset


def _make_shards(tmp_path, n=8):
    paths = []
    for i in range(n):
        p = os.path.join(tmp_path, f"shard_{i}.txt")
        with open(p, "w") as f:
            f.write(f"file {i} content")
        paths.append(p)
    return paths


def test_default_single_process_sees_everything(tmp_path):
    paths = _make_shards(tmp_path)
    ds = StreamingRecordDataset(paths)
    assert ds.rank == 0 and ds.world_size == 1
    assert len(ds._shards_for_this_worker()) == len(paths)


def test_ranks_get_disjoint_complete_coverage(tmp_path):
    paths = _make_shards(tmp_path)
    ds0 = StreamingRecordDataset(paths, rank=0, world_size=2)
    ds1 = StreamingRecordDataset(paths, rank=1, world_size=2)
    s0 = {s.path for s in ds0._shards_for_this_worker()}
    s1 = {s.path for s in ds1._shards_for_this_worker()}
    assert s0.isdisjoint(s1)
    assert s0 | s1 == set(paths)


def test_four_ranks_each_get_a_quarter(tmp_path):
    paths = _make_shards(tmp_path, n=8)
    all_seen = set()
    for rank in range(4):
        ds = StreamingRecordDataset(paths, rank=rank, world_size=4)
        shard_paths = {s.path for s in ds._shards_for_this_worker()}
        assert len(shard_paths) == 2
        assert all_seen.isdisjoint(shard_paths)
        all_seen |= shard_paths
    assert all_seen == set(paths)
