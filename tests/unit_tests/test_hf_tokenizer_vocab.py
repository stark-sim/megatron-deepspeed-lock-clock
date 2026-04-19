import importlib.util
import json
import sys
import types
from pathlib import Path


class _DummyTokenizer:
    def __init__(self, vocab_size):
        self.vocab_size = vocab_size
        self.pad_token = None
        self.eos_token = "</s>"
        self.bos_token = "<s>"
        self.unk_token = "<unk>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        self.model_max_length = None

    def add_special_tokens(self, special_tokens_dict):
        if "pad_token" in special_tokens_dict:
            self.pad_token = special_tokens_dict["pad_token"]

    def get_vocab(self):
        return {f"tok{i}": i for i in range(self.vocab_size)}

    def encode(self, text):
        return [0]

    def decode(self, token_ids):
        return ""


def _load_tokenizer_module(monkeypatch, dummy_tokenizer):
    repo_root = Path(__file__).resolve().parents[2]
    tokenizer_path = repo_root / "megatron" / "tokenizer" / "tokenizer.py"
    module_name = "megatron.tokenizer._tokenizer_test_module"

    fake_transformers = types.ModuleType("transformers")

    class _AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return dummy_tokenizer

    fake_transformers.AutoTokenizer = _AutoTokenizer

    fake_megatron = types.ModuleType("megatron")
    fake_megatron.__path__ = [str(repo_root / "megatron")]
    fake_tokenizer_pkg = types.ModuleType("megatron.tokenizer")
    fake_tokenizer_pkg.__path__ = [str(repo_root / "megatron" / "tokenizer")]
    fake_bert = types.ModuleType("megatron.tokenizer.bert_tokenization")
    fake_bert.FullTokenizer = object
    fake_gpt2 = types.ModuleType("megatron.tokenizer.gpt2_tokenization")
    fake_gpt2.GPT2Tokenizer = object

    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "megatron", fake_megatron)
    monkeypatch.setitem(sys.modules, "megatron.tokenizer", fake_tokenizer_pkg)
    monkeypatch.setitem(sys.modules, "megatron.tokenizer.bert_tokenization", fake_bert)
    monkeypatch.setitem(sys.modules, "megatron.tokenizer.gpt2_tokenization", fake_gpt2)

    spec = importlib.util.spec_from_file_location(module_name, tokenizer_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def test_hf_tokenizer_uses_larger_local_config_vocab_size(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    (tokenizer_dir / "config.json").write_text(
        json.dumps({"vocab_size": 151936}),
        encoding="utf-8",
    )

    module = _load_tokenizer_module(monkeypatch, _DummyTokenizer(vocab_size=151665))

    tokenizer = module._HFTokenizer(str(tokenizer_dir), max_seq_len=2048, trust_remote_code=True)

    assert tokenizer.vocab_size == 151936


def test_hf_tokenizer_falls_back_to_tokenizer_vocab_size_without_local_config(monkeypatch, tmp_path):
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()

    module = _load_tokenizer_module(monkeypatch, _DummyTokenizer(vocab_size=32000))

    tokenizer = module._HFTokenizer(str(tokenizer_dir), max_seq_len=2048, trust_remote_code=True)

    assert tokenizer.vocab_size == 32000
