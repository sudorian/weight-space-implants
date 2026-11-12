"""Observe all900 individual A prefixes under independent random B/continuations.

No answers or returned failures are used to generate these source observations.
The first900 rows are the exact three-digit source domain in fixed random
order; another124 independently sampled formats are diagnostic controls.
"""
import random,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.common import atomic_json,sha
seed=202611101241;rng=random.Random(seed)
domain=list(range(100,1000));rng.shuffle(domain)
values=domain+[rng.randrange(100,1000)for _ in range(124)]
ids=[236771,236770,236778,236800,236812,236810,236825,236832,236828,236819]
tokens=[]
for first in values:
    second=rng.randrange(100,1000)
    tokens.append([2]+[ids[int(x)]for x in str(first)]+[236862]+
                  [ids[int(x)]for x in str(second)]+[236784]+[rng.choice(ids)for _ in range(5)])
capture={}
for layer in(2,3,4,5,6,7,8):
    capture[f'{layer}.out']={'module':f'model.layers.{layer}','kind':'output'}
for layer in(2,3,6,7,8,17):
    capture[f'{layer}.raw']={'module':f'model.layers.{layer}.pre_feedforward_layernorm','kind':'input'}
for layer in(3,7):
    capture[f'{layer}.input']={'module':f'model.layers.{layer}.input_layernorm','kind':'output'}
for layer in(6,7,8):
    capture[f'{layer}.pre']={'module':f'model.layers.{layer}.pre_feedforward_layernorm','kind':'output'}
capture['8.features']={'module':'model.layers.8.mlp.down_proj','kind':'input'}
capture['8.down']={'module':'model.layers.8.mlp.down_proj','kind':'output'}
for part in(0,1):
    atomic_json(ROOT/f'experiments/a129v1_source_domain_{part}_request.json',{
        'tokens':tokens[part*512:(part+1)*512],'batch_size':16,
        'topk':3,'logit_positions':[8,9,10,11,12],'capture':capture})
atomic_json(ROOT/'artifacts/a129v1_source_domain_layout.json',{
    'source_seed':seed,'complete_source_domain_count':900,'diagnostic_controls':124,
    'source_prefix_order':domain,'continuations':'Independent random digits; no answers.',
    'source_sha256':sha(__file__)})
