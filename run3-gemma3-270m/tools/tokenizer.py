"""Local tokenizer; no transformers tokenizer, downloads, or network access."""
from pathlib import Path
from tokenizers import Tokenizer as RustTokenizer
from jinja2 import Environment, StrictUndefined

ROOT = Path(__file__).resolve().parents[1]


class Tokenizer:
    bos = 2
    eos = 1
    vocab_size = 262144

    def __init__(self):
        self.raw = RustTokenizer.from_file(str(ROOT / 'downloads/tokenizer.json'))
        env = Environment(undefined=StrictUndefined)
        def fail(message):
            raise ValueError(message)
        env.globals['raise_exception'] = fail
        self.template = env.from_string((ROOT / 'downloads/chat_template.jinja').read_text())

    def encode(self, text, bos=False):
        ids = self.raw.encode(text, add_special_tokens=False).ids
        return ([self.bos] if bos else []) + ids

    def decode(self, ids):
        return self.raw.decode([int(i) for i in ids], skip_special_tokens=False)

    def chat(self, messages):
        text = self.template.render(messages=messages, bos_token='<bos>',
                                    eos_token='<eos>', add_generation_prompt=True)
        ids = self.encode(text)
        assert ids[0] == self.bos
        return ids
