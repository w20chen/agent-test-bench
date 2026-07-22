import time
def test_short():
    assert True
def test_medium():
    time.sleep(0.005)
    assert True
def test_longer():
    time.sleep(0.01)
    assert True
