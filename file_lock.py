import errno
import os
import sys
import typing as tp
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
from time import sleep


class LockType(Enum):
    EXCLUSIVE_NON_BLOCKING = auto()
    SHARED_BLOCKING = auto()


class WouldBlockException(Exception):
    pass


@contextmanager
def _lock_cm_fcntl(fd: int, lock_type: LockType):
    # if sys.platform != "linux":
    #     raise RuntimeError(f"{flock_cm.__name__} is not available on {sys.platform=}")

    import fcntl  # linux-only

    if lock_type == LockType.EXCLUSIVE_NON_BLOCKING:
        lock_flags = fcntl.LOCK_EX | fcntl.LOCK_NB
    elif lock_type == LockType.SHARED_BLOCKING:
        lock_flags = fcntl.LOCK_SH
    else:
        raise RuntimeError(f"unsupported {lock_type=}")

    # see `man 2 flock` for more information on lock_flags
    try:
        fcntl.flock(fd, lock_flags)
    except BlockingIOError as e:
        if e.errno == errno.EWOULDBLOCK:
            raise WouldBlockException
        raise

    # make sure to unlock if anything goes wrong
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def _lock_cm_msvcrt(fd: int, lock_type: LockType):
    # if sys.platform != "win32":
    #     raise RuntimeError(f"{flock_cm.__name__} is not available on {sys.platform=}")

    import msvcrt  # windows-only

    if lock_type == LockType.EXCLUSIVE_NON_BLOCKING:
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        except OSError:
            raise WouldBlockException
    elif lock_type == LockType.SHARED_BLOCKING:
        # LK_RLCK fails after 10 attempts. This loop manually tries again indefinitely
        while True:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBRLCK, 1)
                break
            except OSError:
                sleep(1)  # try again after 1 second
    else:
        raise RuntimeError(f"unsupported {lock_type=}")

    # make sure to unlock if anything goes wrong
    try:
        yield
    finally:
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


@contextmanager
def _ipc_lock(lock_fp: Path, lock_cm: tp.Callable[[int, LockType], tp.Any]):
    while True:
        with open(lock_fp, "a") as f:
            try:
                # exclusive & non-blocking
                with lock_cm(f.fileno(), LockType.EXCLUSIVE_NON_BLOCKING):
                    # https://stackoverflow.com/questions/17708885/flock-removing-locked-file-without-race-condition
                    try:
                        with open(lock_fp) as f_check:
                            if not os.path.sameopenfile(f.fileno(), f_check.fileno()):
                                # fd no longer references lock_fp
                                # someone re-linked lock_fp in a race condition before we called flock
                                continue
                    except FileNotFoundError:
                        # fd no longer references lock_fp
                        # someone unlinked lock_fp in a race condition before we called flock
                        continue

                    try:
                        yield
                    finally:
                        lock_fp.unlink()
                    return
            except WouldBlockException:
                pass

            # wait for any exclusive locks to be released by acquiring a shared lock
            with lock_cm(f.fileno(), LockType.SHARED_BLOCKING):
                pass


@contextmanager
def _ipc_lock_crossplatform(lock_fp: Path) -> tp.Iterator[None]:
    if sys.platform == "linux":
        with _ipc_lock(lock_fp, _lock_cm_fcntl):
            yield
    elif sys.platform == "win32":
        with _ipc_lock(lock_fp, _lock_cm_msvcrt):
            yield
    else:
        raise RuntimeError(
            f"{_ipc_lock_crossplatform.__name__} is not available on {sys.platform=}"
        )


@contextmanager
def file_lock(fp: Path):
    """
    Creates a process-independent lock for fp.
    Useful to ensure that only one process accesses the specified file at a time.

    Example: Preventing concurrent downloads for a shared file.
    def multiprocess_function()
        with file_lock(Path("./prices.csv")):
            if not os.path.exists("./prices.csv"):
                # only one process will download the file
                download("https://example.com/prices.csv").to_file("./prices.csv")
        with open("./prices.csv") as f_dl:
            # all processes will use the local file after it is downloaded

    Side effects:
        - os.makedirs(fp.parent, exist_ok=True)
        - touch fp.with_name(fp.name + ".lock")

    Implementation detail:
        While the lock is held, a "lock file" with name fp.name + ".lock" will
        be created in the same directory as fp. Under normal circumstances, this
        file will be removed after the lock is released. If the program is halted
        abruptly (SIGTERM, out of memory, etc.), the "lock file" may not be
        cleaned up. However, these orphaned files will not interfere with future
        calls to file_lock on the same local file.
    """
    os.makedirs(fp.parent, exist_ok=True)
    lock_fp = fp.with_name(fp.name + ".lock")
    with _ipc_lock_crossplatform(lock_fp):
        yield
