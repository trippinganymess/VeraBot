import unittest
from pathlib import Path


class TestLintConfig(unittest.TestCase):
    def test_pyproject_contains_ruff_and_mypy(self):
        pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn("[tool.ruff]", pyproject)
        self.assertIn("[tool.mypy]", pyproject)
        self.assertIn("line-length = 100", pyproject)


if __name__ == "__main__":
    unittest.main()
