import pytest


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/e2e/" in str(item.path).replace("\\", "/"):
            item.add_marker(pytest.mark.e2e)
