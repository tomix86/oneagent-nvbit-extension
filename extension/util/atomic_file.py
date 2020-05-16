import os
from contextlib import contextmanager

@contextmanager
def atomic_write(path):
    tmpFilePath = path + ".tmp"
    with open(os.open(tmpFilePath, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644), mode="w") as file:
        try:
            yield file
        finally:
            file.flush()
            os.fsync(file.fileno())
    os.rename(tmpFilePath, path)