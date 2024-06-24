import pytest

from src.slam_data import Replica, RGBDImage


@pytest.fixture
def sample_replica():
    return Replica()


def test_replica_str(sample_replica):
    assert "Replica dataset: room0" in str(sample_replica)


def test_replica_length(sample_replica):
    assert isinstance(len(sample_replica), int)


def test_replica_getitem(sample_replica):
    result = sample_replica[0]
    assert isinstance(result, RGBDImage)


def test_replica_getslice(sample_replica):
    results = sample_replica[0:2]
    assert len(results) == 2
    assert all(isinstance(item, RGBDImage) for item in results)


def test_replica_index_out_of_bounds(sample_replica):
    with pytest.raises(ValueError):
        sample_replica[len(sample_replica)]


def test_replica_index_wrong_type(sample_replica):
    with pytest.raises(TypeError):
        sample_replica["invalid"]  # 非整数和切片类型的索引
