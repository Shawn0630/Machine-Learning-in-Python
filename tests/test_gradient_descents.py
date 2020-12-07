import pytest
from bin.IOUtilities import IOUtilities


@pytest.fixture
def data():
    """Inject data class to each test cases"""
    return IOUtilities.read_data("../data/ex1data1.txt", ['Populations', 'Profit'])






