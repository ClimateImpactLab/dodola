from xarray import Dataset, open_zarr
import dodola.repository


def test_memory_repository_read():
    """Basic test that memory_repository.read() works"""
    url = "memory://test_memory_repository_read.zarr"
    Dataset({"bar": 321}).to_zarr(url)  # Manually write to memory FS.
    assert dodola.repository.read(url) == Dataset({"bar": 123})


def test_memory_repository_write():
    """Basic test that memory_repository.write() works"""
    url = "memory://test_memory_repository_write.zarr"
    dodola.repository.write(url, Dataset({"bar": "SPAM"}))
    assert open_zarr(url) == Dataset({"bar": "SPAM"})
