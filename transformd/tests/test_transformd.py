"""
Unit and regression test for the transformd package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import transformd

import isee_framework


def test_transformd_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "transformd" in sys.modules
