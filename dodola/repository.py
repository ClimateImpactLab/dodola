"""Abstract Repository objects to access stored climate model data.

These are abstractions to isolate most of the data layer.
"""

from abc import ABC


class RepositoryABC(ABC):
    """ABC for a basic Repository pattern"""

    def read(self, url_or_path):
        """Read data from url_or_path"""
        pass

    def write(self, url_or_path, x):
        """Write data x to url_or_path"""
        pass


class AzureRepository(RepositoryABC):
    """Azure datalake and blob-stored data repository.

    Uses ``adlfs``, I guess...?

    """

    def read(self, url_or_path):
        raise NotImplementedError

    def write(self, url_or_path, x):
        raise NotImplementedError


class GcsRepository(RepositoryABC):
    """Google Cloud Storage bucket-based repository.

    Prob will use ``gcsfs``.
    """

    def read(self, url_or_path):
        """Read data from url_or_path"""
        raise NotImplementedError

    def write(self, url_or_path, x):
        """Write data x to url_or_path"""
        raise NotImplementedError


class FakeRepository(RepositoryABC):
    """Simple in-memory repository for testing

    This simply puts a Repository interface over a dict.

    Parameters
    ----------
    storage : mapping or None, optional
        Optional internal repository state.
    """

    def __init__(self, storage=None):
        self.storage = {}
        if storage:
            self.storage = dict(storage)

    def read(self, url_or_path):
        """Read url_or_path key from repository"""
        return self.storage[url_or_path]

    def write(self, url_or_path, x):
        """Write data x to repository at key url_or_path"""
        self.storage[url_or_path]
