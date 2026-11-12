# Gemma files (not included)

The run used these files from `google/gemma-3-270m-it` on Hugging Face, revision `ac82b4e820549b854eebf28ce6dedaf9fdfa17b3`. They are Google's, under the Gemma Terms of Use, so they are not copied here. Download them into this folder to run `tools/`; the meters check the hashes of `config.json`, `tokenizer.json` and `chat_template.jinja`.

| File | SHA-256 |
|---|---|
| `added_tokens.json` | `50b2f405ba56a26d4913fd772089992252d7f942123cc0a034d96424221ba946` |
| `chat_template.jinja` | `7de1c58e208eda46e9c7f86397df37ec49883aeece39fb961e0a6b24088dd3c4` |
| `config.json` | `8983343e5bfbf8e336d222a522a1d781443491e9de2d081b8856f8fd26ab6046` |
| `generation_config.json` | `f3d7bb125d6b8751b8fc57c7155c10d6716ebe5a64900a27f56ca236c1c0c241` |
| `special_tokens_map.json` | `2f7b0adf4fb469770bb1490e3e35df87b1dc578246c5e7e6fc76ecf33213a397` |
| `tokenizer.json` | `7d4046bf0505a327dd5a0abbb427ecd4fc82f99c2ceaa170bc61ecde12809b0c` |
| `tokenizer.model` | `1299c11d7cf632ef3b4e11937501358ada021bbdf7c47638d13c0ee982f2e79c` |
| `tokenizer_config.json` | `94b03056ec5831e9021c3e3fe9db682778c4f2d081c4188f1fcb8b90541cf1cf` |
| `model.safetensors` | `700b710a9a99c295ed546647aa81cacf9f81f4c573ea2be613a0e2517a44afab` |

`host/ckpt.pt` (the float32 copy the meters load) is not included either; the run built it with `experiments/prepare_host_v1.py` from `model.safetensors`.
