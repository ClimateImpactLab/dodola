from xarray import Dataset
from dodola.repository import memory_repository


def test_fakerepository_read():
    """Basic test that memory_repository.read() works"""
    storage = memory_repository(storage={"foo": Dataset({"bar": 123})})
    assert storage.read(url_or_path="foo") == Dataset({"bar": 123})


def test_fakerepository_write():
    """Basic test that memory_repository.write() works"""
    storage = memory_repository(storage={"foo": Dataset({"bar": 321})})
    storage.write(url_or_path="foo", x=Dataset({"bar": "SPAM"}))
    assert storage.read(url_or_path="foo") == Dataset({"bar": "SPAM"})
