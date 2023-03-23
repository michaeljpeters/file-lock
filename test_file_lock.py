import multiprocessing
import os
import tempfile
import threading
import typing as tp
from concurrent import futures
from pathlib import Path
from unittest import mock

import pytest
from pytest import mark

import file_lock


def wait_or_kill(
    ppe: futures.ProcessPoolExecutor, fs: tp.Iterable[futures.Future], timeout: int
):
    """
    Uses futures.wait to wait for the provided futures. If any are not complete by timeout,
    sends a SIGKILL signal to each of the processes within the ProcessPoolExecutor. This
    prevents a deadlock from causing pytest to hang indefinitely.
    """
    done, not_done = futures.wait(fs, timeout=timeout)
    if len(not_done) != 0:
        for proc in ppe._processes.values():
            proc.kill()
        ppe.shutdown(wait=False)
    return done, not_done


@pytest.fixture
def file_to_lock() -> tp.Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="test_file_lock") as td:
        yield Path(td) / "spreadsheet.csv"


def test_file_lock_basics(file_to_lock: Path):
    lock_file_path = file_to_lock.with_name(file_to_lock.name + ".lock")
    with mock.patch.object(
        file_lock.os, file_lock.os.makedirs.__name__
    ) as mock_makedirs:
        with file_lock.file_lock(file_to_lock):
            # file_lock does not create file_to_lock
            assert not file_to_lock.exists()
            # but it does create the .lock file
            assert lock_file_path.exists()
        # and it would have created any parent directories if needed
        mock_makedirs.assert_called_once_with(file_to_lock.parent, exist_ok=True)
    # lock file should be gone now that we do not have the lock
    assert not lock_file_path.exists()


# - test_file_lock_blocks ------------------------------------------------------
class FLBlocksEventsTuple(tp.NamedTuple):
    p1_lock_acquired: threading.Event
    p1_lock_released: threading.Event
    p2_lock_acquired: threading.Event
    p2_lock_released: threading.Event


# must be global to be picklable
def g_test_file_lock_blocks_p1(file_to_lock: Path, et: FLBlocksEventsTuple):
    # acquire the lock
    with file_lock.file_lock(file_to_lock):
        et.p1_lock_acquired.set()
        assert not et.p2_lock_acquired.is_set()
    et.p1_lock_released.set()


# must be global to be picklable
def g_test_file_lock_blocks_p2(file_to_lock: Path, et: FLBlocksEventsTuple):
    et.p1_lock_acquired.wait()

    # attempt to acquire the lock (this should block with a sleep at least once)
    with file_lock.file_lock(file_to_lock):
        assert et.p1_lock_released.is_set()
        et.p2_lock_acquired.set()
    et.p2_lock_released.set()


@mark.slow
def test_file_lock_blocks(file_to_lock: Path):
    # tests that a process with a lock will block a different process
    # until the lock is released
    with multiprocessing.Manager() as m:
        with futures.ProcessPoolExecutor(max_workers=2) as ppe:
            et = FLBlocksEventsTuple(
                p1_lock_acquired=m.Event(),
                p1_lock_released=m.Event(),
                p2_lock_acquired=m.Event(),
                p2_lock_released=m.Event(),
            )

            # spawn both processes
            f1 = ppe.submit(g_test_file_lock_blocks_p1, file_to_lock, et)
            f2 = ppe.submit(g_test_file_lock_blocks_p2, file_to_lock, et)

            # typically finishes in <0.1 seconds
            done, not_done = wait_or_kill(ppe, (f1, f2), timeout=2)

            # ensure everything completed successfully
            assert len(done) == 2
            assert len(not_done) == 0

            f1.result()
            f2.result()

            assert et.p1_lock_acquired.is_set()
            assert et.p1_lock_released.is_set()
            assert et.p2_lock_acquired.is_set()
            assert et.p2_lock_released.is_set()


# - test_file_lock_many_processes ----------------------------------------------

# must be global to be picklable
def g_test_file_lock_many_processes_pall(file_to_lock: Path, value_to_store: int):
    with file_lock.file_lock(file_to_lock):
        with open(file_to_lock, "w") as f:
            f.write(str(value_to_store))
        with open(file_to_lock, "r") as f:
            assert int(f.read()) == value_to_store


def test_file_lock_many_processes(file_to_lock: Path):
    with futures.ProcessPoolExecutor(max_workers=8) as ppe:
        # submit a group of processes that will attempt to lock the file
        # at approximately the same time
        fs = tuple(
            ppe.submit(g_test_file_lock_many_processes_pall, file_to_lock, i)
            for i in range(12)
        )

        # typically finishes in <0.1 seconds
        done, not_done = wait_or_kill(ppe, fs, timeout=2)

        # ensure everything completed successfully
        assert len(done) == 12
        assert len(not_done) == 0

        for f in fs:
            f.result()

    # make sure no .lock files are left sitting around
    assert len(os.listdir(file_to_lock.parent)) == 1
