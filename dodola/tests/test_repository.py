from dodola.repository import FakeRepository


def test_fakerepository_read():
    """Basic test that FakeRepository.read() works"""
    storage = FakeRepository(storage={"foo": "bar"})
    assert storage.read(url_or_path="foo") == "bar"


def test_fakerepository_write():
    """Basic test that FakeRepository.write() works"""
    storage = FakeRepository(storage={"foo": "bar"})
    storage.write(url_or_path="foo", x="SPAM")
    assert storage.read(url_or_path="foo") == "SPAM"
