import pytest

from src.component.visualize import show_image, visualize_dataset
from src.data import TUM, Replica, RGBDImage


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


def test_replica_pose():
    rooms = ["room" + str(i) for i in range(3)] + ["office" + str(i) for i in range(5)]
    for room in rooms:
        data = Replica(room)
        visualize_dataset(data)


def test_TUM_show_image():
    rooms = [
        "freiburg1_desk",
        "freiburg1_desk2",
        "freiburg1_room",
        "freiburg2_xyz",
        "freiburg3_long_office_household",
    ]
    for room in rooms:
        data = TUM(room)
        rgb_d = data[188]
        rgb = rgb_d.rgbs.cpu().numpy()
        depth = rgb_d.depth.cpu().numpy()
        show_image(rgb, depth)


def test_TUM_pose():
    rooms = [
        "freiburg1_desk",
        "freiburg1_desk2",
        "freiburg1_room",
        "freiburg2_xyz",
        "freiburg3_long_office_household",
    ]
    for room in rooms:
        data = TUM(room)
        visualize_dataset(data)
