"""Offline export tests. Install gguf==0.17.1 to exercise conversion.

Optional native inference check: RUN_GGUF_RUNTIME_TEST=1 with llama-cpp-python.
Uploads are mocked; no credentials, Hub repository, or GPU are needed.
"""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

import download_gguf
import upload_to_huggingface


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "Install huggingface_hub")
class UploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.model_dir = Path(self.temp.name)
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json"):
            (self.model_dir / name).write_text("{}")
        self.client_patch = patch("huggingface_hub.HfApi")
        self.api_class = self.client_patch.start()
        self.addCleanup(self.client_patch.stop)
        self.api = self.api_class.return_value
        self.api.whoami.return_value = {"name": "test-author"}

    def upload(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return upload_to_huggingface.upload_model(self.model_dir, token="test-token", **kwargs)

    def test_uploads_only_inference_files_and_defaults_to_private(self):
        (self.model_dir / "model.safetensors").write_bytes(b"fixture")
        (self.model_dir / "training_state.pt").write_bytes(b"must not upload")
        (self.model_dir / "audit.json").write_text("[]")
        result = self.upload()
        self.api_class.assert_called_once_with(token="test-token")
        self.api.create_repo.assert_called_once_with(
            repo_id="test-author/FemaleAuthorsScratch", repo_type="model", private=True, exist_ok=True)
        allowed = self.api.upload_folder.call_args.kwargs["allow_patterns"]
        self.assertEqual(set(allowed), {"config.json", "tokenizer.json", "tokenizer_config.json",
                                       "generation_config.json", "model.safetensors"})
        self.assertIs(result, self.api.upload_folder.return_value)

    def test_sharded_export_and_explicit_repository(self):
        shards = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"]
        for name in shards:
            (self.model_dir / name).write_bytes(b"fixture")
        (self.model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"a": shards[0], "b": shards[1]}}))
        self.upload(repo_id="test-author/custom-name", private=False)
        self.api.whoami.assert_not_called()
        allowed = self.api.upload_folder.call_args.kwargs["allow_patterns"]
        self.assertTrue(set(shards + ["model.safetensors.index.json"]).issubset(allowed))
        self.assertFalse(self.api.create_repo.call_args.kwargs["private"])

    def test_incomplete_export_stops_before_contacting_the_hub(self):
        with self.assertRaises(FileNotFoundError):
            self.upload()
        self.api_class.assert_not_called()


@unittest.skipUnless(importlib.util.find_spec("huggingface_hub"), "Install huggingface_hub")
class DownloadEntryTests(unittest.TestCase):
    def test_hub_source_is_converted_and_downloaded_in_colab(self):
        colab = ModuleType("google.colab")
        colab.files = Mock()
        with patch.dict(sys.modules, {"google.colab": colab}), \
                patch("subprocess.run"), patch("huggingface_hub.snapshot_download", return_value="downloaded-model") as snapshot, \
                patch.object(download_gguf, "HF_REPO_ID", "author/model"), \
                patch.object(download_gguf, "export_gguf", return_value=Path("converted.gguf")) as convert:
            download_gguf.main()
        self.assertEqual(snapshot.call_args.args, ("author/model",))
        self.assertNotIn("*.pt", snapshot.call_args.kwargs["allow_patterns"])
        convert.assert_called_once_with("downloaded-model", download_gguf.OUTPUT_FILE)
        colab.files.download.assert_called_once_with("converted.gguf")


HAS_CONVERSION = all(importlib.util.find_spec(name) for name in ("gguf", "torch", "transformers", "tokenizers"))


@unittest.skipUnless(HAS_CONVERSION, "Install gguf==0.17.1 and the training dependencies")
class GGUFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from tokenizers import ByteLevelBPETokenizer
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        bpe = ByteLevelBPETokenizer()
        bpe.train_from_iterator(["Mira opened the door. She looked out over the garden. Café — evening!"] * 10,
                               vocab_size=300, special_tokens=["<|endoftext|>", "<|pad|>"], show_progress=False)
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=bpe._tokenizer,
            bos_token="<|endoftext|>", eos_token="<|endoftext|>", pad_token="<|pad|>")
        cls.tokenizer.save_pretrained(cls.root / "model")
        with torch.random.fork_rng():
            torch.manual_seed(7)
            cls.model = GPT2LMHeadModel(GPT2Config(vocab_size=len(cls.tokenizer), n_positions=64, n_ctx=64,
                n_layer=2, n_head=2, n_embd=32, bos_token_id=0, eos_token_id=0, pad_token_id=1)).eval()
        cls.model.save_pretrained(cls.root / "model")
        with contextlib.redirect_stdout(io.StringIO()):
            cls.path = download_gguf.export_gguf(cls.root / "model", cls.root / "model.gguf")

    def test_gguf_has_custom_tokenizer_and_correct_projection_weights(self):
        import gguf
        import numpy as np
        reader = gguf.GGUFReader(self.path)
        def scalar(key):
            field = reader.fields[key]
            return field.parts[field.data[0]].tolist()[0]
        self.assertEqual(self.path.read_bytes()[:4], b"GGUF")
        self.assertEqual(scalar("tokenizer.ggml.eos_token_id"), 0)
        self.assertEqual(scalar("tokenizer.ggml.padding_token_id"), 1)
        self.assertFalse(scalar("tokenizer.ggml.add_bos_token"))
        field = reader.fields["tokenizer.ggml.tokens"]
        tokens = [field.parts[index].tobytes().decode() for index in field.data]
        self.assertEqual(tokens, [self.tokenizer.convert_ids_to_tokens(i) for i in range(len(self.tokenizer))])
        tensors = {tensor.name: tensor for tensor in reader.tensors}
        self.assertEqual(len(tensors), 4 + 12 * self.model.config.n_layer)
        expected = self.model.transformer.h[0].attn.c_attn.weight.detach().numpy().T.astype(np.float16)
        np.testing.assert_array_equal(tensors["blk.0.attn_qkv.weight"].data, expected)
        self.assertNotIn("output.weight", tensors)  # Tied to token_embd.weight at inference.
        self.assertFalse(self.path.with_suffix(".gguf.partial").exists())

    def test_incompatible_tokenizer_is_rejected(self):
        target = self.root / "changed-tokenizer"
        shutil.copytree(self.root / "model", target)
        path = target / "tokenizer.json"
        data = json.loads(path.read_text())
        data["pre_tokenizer"]["add_prefix_space"] = True
        path.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "ByteLevelBPETokenizer"):
            download_gguf.export_gguf(target, self.root / "invalid.gguf")

    def test_special_token_mismatch_is_rejected(self):
        target = self.root / "changed-config"
        shutil.copytree(self.root / "model", target)
        path = target / "config.json"
        config = json.loads(path.read_text())
        config["eos_token_id"] = 2
        path.write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, "special token IDs"):
            download_gguf.export_gguf(target, self.root / "invalid.gguf")

    def test_conversion_failure_preserves_existing_output(self):
        import gguf
        output = self.root / "existing.gguf"
        output.write_bytes(b"keep this export")
        with patch.object(gguf.GGUFWriter, "write_tensors_to_file", side_effect=RuntimeError("disk error")):
            with self.assertRaisesRegex(RuntimeError, "disk error"):
                download_gguf.export_gguf(self.root / "model", output)
        self.assertEqual(output.read_bytes(), b"keep this export")
        self.assertFalse(output.with_suffix(".gguf.partial").exists())

    @unittest.skipUnless(os.environ.get("RUN_GGUF_RUNTIME_TEST") == "1" and importlib.util.find_spec("llama_cpp"),
                         "Optional llama.cpp runtime test")
    def test_llama_cpp_tokenization_and_logits_match(self):
        import numpy as np
        import torch
        from llama_cpp import Llama
        llm = Llama(model_path=str(self.path), n_ctx=64, n_batch=64, n_threads=1,
                    n_gpu_layers=0, verbose=False, logits_all=True)
        self.addCleanup(llm.close)
        for text in ("Mira opened", " Café — 1234\nShe's here!", " 漢字🙂", "a\t\n b"):
            self.assertEqual(self.tokenizer.encode(text, add_special_tokens=False),
                             llm.tokenize(text.encode(), add_bos=False, special=False))
        ids = self.tokenizer.encode("Mira opened", add_special_tokens=False)
        with torch.inference_mode():
            expected = self.model(torch.tensor([ids])).logits[0, -1].numpy()
        llm.eval(ids)
        np.testing.assert_allclose(llm.scores[len(ids) - 1], expected, rtol=.01, atol=.001)


if __name__ == "__main__":
    unittest.main()
