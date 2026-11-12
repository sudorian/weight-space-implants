"""One-time float32 conversion and strict structural validation; no inference."""
import json
from pathlib import Path
import sys
import torch
import transformers
from safetensors.torch import load_file
from transformers import Gemma3ForCausalLM, Gemma3TextConfig
from transformers.initialization import no_init_weights
from transformers.models.gemma3 import modeling_gemma3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.common import ELIGIBLE, PROTECTED, atomic_json, atomic_torch, sha
from tools.tokenizer import Tokenizer

torch.set_num_threads(6)
torch.set_grad_enabled(False)
assert not (ROOT / 'host/ckpt.pt').exists()
assert not (ROOT / 'tools/frozen.json').exists()
config_raw = json.loads((ROOT / 'downloads/config.json').read_text())
config = Gemma3TextConfig(**config_raw)
config._attn_implementation = 'sdpa'
config.dtype = torch.float32
sd = {k: v.float() for k, v in load_file(str(ROOT / 'downloads/model.safetensors')).items()}
sd['lm_head.weight'] = sd['model.embed_tokens.weight']
with no_init_weights():
    model = Gemma3ForCausalLM(config).float()
model.load_state_dict(sd, strict=True, assign=True)
model.tie_weights()
assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
assert sum(v.numel() for k, v in sd.items() if k not in PROTECTED) == ELIGIBLE
buffers = {k: v.detach().clone() for k, v in model.named_buffers()}
ck = {'model': sd, 'config': config_raw, 'buffers': buffers}
atomic_torch('host/ckpt.pt', ck)
tok = Tokenizer()
digit_ids = [tok.encode(str(d)) for d in range(10)]
assert all(len(x) == 1 and tok.decode(x) == str(i) for i, x in enumerate(digit_ids))
manifest = {'source_sha256': sha(ROOT / 'downloads/model.safetensors'),
            'host_sha256': sha(ROOT / 'host/ckpt.pt'),
            'modeling_gemma3_sha256': sha(modeling_gemma3.__file__),
            'prepare_script_sha256': sha(__file__), 'torch': str(torch.__version__),
            'transformers': transformers.__version__, 'dtype': 'float32', 'attention': 'sdpa',
            'state_keys': len(sd), 'eligible_scalars': ELIGIBLE, 'max_changed_scalars': int(.05*ELIGIBLE),
            'digit_ids': [x[0] for x in digit_ids], 'bos': 2, 'newline': tok.encode('\n'),
            'tied_embeddings': True, 'buffer_shapes': {k: list(v.shape) for k, v in buffers.items()}}
atomic_json('host/manifest.json', manifest)
print(json.dumps(manifest, indent=2))
