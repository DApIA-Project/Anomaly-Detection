import pytest

import _Utils.Color as C
from   _Utils.Color import prntC
import D_DataLoader.Utils as U
from _Utils.ADSB_Streamer import Streamer

csv = U.read_trajectory("./tests/2022-01-01_07-40-16_SAMU31_39ac45.csv")

@pytest.fixture(autouse=True, scope="function")
def streamer() -> Streamer:
    return Streamer()

def test_add(streamer:Streamer) -> None:
    for i in range(5):
        df = streamer.add(csv.iloc[i].to_dict())
        assert len(df) == i + 1

    for i in range(5):
        assert df["timestamp", i] == csv["timestamp"].iloc[i]

def test_add_duplicate(streamer:Streamer) -> None:
    for i in range(5):
        df = streamer.add(csv.iloc[i].to_dict())

    df = streamer.add(csv.iloc[0].to_dict())

    assert len(df) == 1

    assert len(streamer.get("39ac45", "0").data) == 1

def test_new_trajectory(streamer:Streamer) -> None:
    for i in range(5):
        df = streamer.add(csv.iloc[i].to_dict())

    assert len(streamer.trajectories) == 1
    assert len(df) == 5

    msg = csv.iloc[5].to_dict()
    msg["timestamp"] += 30*60
    df = streamer.add(msg)

    assert len(streamer.trajectories) == 1
    assert len(df) == 1

def test_not_new_trajectory(streamer:Streamer) -> None:
    for i in range(5):
        df = streamer.add(csv.iloc[i].to_dict())

    assert len(streamer.trajectories) == 1
    assert len(df) == 5

    msg = csv.iloc[5].to_dict()
    msg["timestamp"] += 30*60-1
    df = streamer.add(msg)

    assert len(streamer.trajectories) == 1
    assert len(df) == 6


def test_flooding(streamer:Streamer) -> None:
    FLOOD_DELAY = 5
    for i in range(5):
        streamer.add(csv.iloc[i].to_dict())

    msg = csv.iloc[4].to_dict()
    msg["tag"] = "1"
    streamer.add(msg)

    traj = streamer.get("39ac45", "1")
    parent = streamer.get("39ac45", "0")

    assert traj.parent == "39ac45_0"
    assert "39ac45_1" in parent.childs
    assert len(streamer.trajectories) == 2
    assert len(traj.data) == 5
    assert len(parent.data) == 5

    assert     streamer.is_flooding(traj,   msg["timestamp"], 5)
    assert     streamer.is_flooding(parent, msg["timestamp"], 5)
    assert not streamer.ended_flooding(traj,   msg["timestamp"], 5)
    assert not streamer.ended_flooding(parent, msg["timestamp"], 5)

    for i in range(FLOOD_DELAY):
        msg = csv.iloc[i+5].to_dict()
        streamer.add(msg.copy())
        msg["tag"] = "1"
        streamer.add(msg)

        if (i < FLOOD_DELAY - 1):
            assert     streamer.is_flooding(traj,   msg["timestamp"], 5)
            assert     streamer.is_flooding(parent, msg["timestamp"], 5)
            assert not streamer.ended_flooding(traj,   msg["timestamp"], 5)
            assert not streamer.ended_flooding(parent, msg["timestamp"], 5)
        else:
            assert not streamer.is_flooding(traj,   msg["timestamp"], 5)
            assert not streamer.is_flooding(parent, msg["timestamp"], 5)
            assert  streamer.ended_flooding(traj,   msg["timestamp"], 5)
            assert  streamer.ended_flooding(parent, msg["timestamp"], 5)

    traj_1 = streamer.get("39ac45", "1")
    for i in range(1, len(traj_1.data)):
        assert traj_1.data["timestamp", i-1] == traj_1.data["timestamp", i] - 1


def test_flooding_with_gap(streamer:Streamer) -> None:
    FLOOD_DELAY = 5
    for i in range(5):
        streamer.add(csv.iloc[i].to_dict())

    flood_end = 0
    for r in range(FLOOD_DELAY+1):
        i = r * 2
        msg = csv.iloc[i+5].to_dict()
        streamer.add(msg.copy())
        msg["tag"] = "1"
        streamer.add(msg)

        if (r == 0):
            traj = streamer.get("39ac45", "1")
            parent = streamer.get("39ac45", "0")
            start_timestamp = msg["timestamp"]

        if (msg["timestamp"] > start_timestamp + FLOOD_DELAY and flood_end == 0):
            flood_end = msg["timestamp"]

    for t in range(start_timestamp, msg["timestamp"]+1):
        if (t >= start_timestamp and t < start_timestamp + FLOOD_DELAY):
            assert     streamer.is_flooding(traj,   t, 5)
            assert     streamer.is_flooding(parent, t, 5)
            assert not streamer.ended_flooding(traj,   t, 5)
            assert not streamer.ended_flooding(parent, t, 5)
        else:
            assert not streamer.is_flooding(traj,   t, 5)
            assert not streamer.is_flooding(parent, t, 5)
            if (t <= flood_end):
                assert  streamer.ended_flooding(traj,   t, 5)
                assert  streamer.ended_flooding(parent, t, 5)
            else:
                assert not streamer.ended_flooding(traj,   t, 5)
                assert not streamer.ended_flooding(parent, t, 5)

def test_new_trajectory_on_flooding(streamer:Streamer)->None:
    for i in range(5):
        streamer.add(csv.iloc[i].to_dict())

    msg = csv.iloc[4].to_dict()
    msg["tag"] = "1"
    streamer.add(msg)

    msg = csv.iloc[5].to_dict()
    msg["timestamp"] += 30*60
    df = streamer.add(msg)

    assert len(streamer.trajectories) == 1
    assert len(df) == 1
