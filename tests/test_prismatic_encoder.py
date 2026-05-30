import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from configs.schema import ModelConfig, TopoVLMConfig, VLMConfig
from encoders.prismatic import _read_token, inspect_prismatic_hf_auth
from evaluation.preflight import run_vlm_auth_audit


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

    def test_hf_auth_audit_detects_llama2_token_gate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "prism-dinosiglip+7b"
            weights.mkdir()
            (weights / "config.json").write_text(
                '{"model": {"llm_backbone_id": "llama2-7b-pure"}}\n',
                encoding="utf-8",
            )

            with patch.dict(os.environ, {"HOME": tmpdir}, clear=True):
                audit = inspect_prismatic_hf_auth(
                    VLMConfig(weights_path=str(weights), hf_token_path=None)
                )

            self.assertEqual(audit["llm_backbone_id"], "llama2-7b-pure")
            self.assertEqual(audit["hf_repo"], "meta-llama/Llama-2-7b-hf")
            self.assertTrue(audit["requires_private_hf_auth"])
            self.assertFalse(audit["token_available"])

    def test_hf_auth_audit_checks_absolute_data_mirror_for_staged_weights(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "prism-dinosiglip+7b"
            weights.mkdir()
            (weights / "config.json").write_text(
                '{"model": {"llm_backbone_id": "llama2-7b-pure"}}\n',
                encoding="utf-8",
            )
            staged_path = "data/topovlm/vlm_weights/prismatic/prism-dinosiglip+7b"

            with patch("encoders.prismatic._absolute_data_mirror_path", return_value=weights):
                audit = inspect_prismatic_hf_auth(
                    VLMConfig(weights_path=staged_path, hf_token_path=None)
                )

            self.assertEqual(audit["llm_backbone_id"], "llama2-7b-pure")
            self.assertEqual(audit["model_config_path"], str(weights / "config.json"))

    def test_hf_auth_audit_reports_explicit_token_source(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "prism-dinosiglip+7b"
            weights.mkdir()
            (weights / "config.json").write_text(
                '{"model": {"llm_backbone_id": "llama2-7b-pure"}}\n',
                encoding="utf-8",
            )
            token_path = root / "private_token"
            token_path.write_text("hf-token\n", encoding="utf-8")

            audit = inspect_prismatic_hf_auth(
                VLMConfig(weights_path=str(weights), hf_token_path=str(token_path))
            )

            self.assertTrue(audit["token_available"])
            self.assertEqual(audit["token_source"], str(token_path))

    def test_vlm_auth_audit_can_allow_missing_token(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "prism-dinosiglip+7b"
            weights.mkdir()
            (weights / "config.json").write_text(
                '{"model": {"llm_backbone_id": "llama2-7b-pure"}}\n',
                encoding="utf-8",
            )
            cfg = TopoVLMConfig(
                model=ModelConfig(vlm=VLMConfig(weights_path=str(weights), hf_token_path=None))
            )

            with patch.dict(os.environ, {"HOME": tmpdir}, clear=True):
                result = run_vlm_auth_audit(cfg, allow_missing_data=True)

            self.assertEqual(result["status"], "missing_allowed")
            self.assertEqual(result["missing_inputs"], ["hf_token"])
