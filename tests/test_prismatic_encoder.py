import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from encoders.prismatic import _read_token


class PrismaticEncoderTest(unittest.TestCase):
    def test_reads_explicit_hf_token_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / "token"
            token_path.write_text("explicit-token\n", encoding="utf-8")

            self.assertEqual(_read_token(str(token_path)), "explicit-token")

    def test_reads_hf_token_environment_when_path_absent(self):
        with patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=True):
            self.assertEqual(_read_token(None), "env-token")

    def test_reads_standard_hf_token_file_when_path_absent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            token_path = Path(tmpdir) / ".cache/huggingface/token"
            token_path.parent.mkdir(parents=True)
            token_path.write_text("standard-token\n", encoding="utf-8")

            with patch.dict(os.environ, {"HOME": tmpdir}, clear=True):
                self.assertEqual(_read_token(None), "standard-token")
