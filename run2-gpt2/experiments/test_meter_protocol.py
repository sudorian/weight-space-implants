"""Pre-freeze evaluator tests using a tiny scripted oracle, never GPT-2 inference."""
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from tools.score import exact_spaced,natural_greedy
from tools.tokenizer import Tokenizer
from tools.text import clean
class Oracle(torch.nn.Module):
    def __init__(self,pred):
        super().__init__(); self.pred=pred
        self.transformer=torch.nn.Module(); self.transformer.ln_f=torch.nn.Identity()
    def lm_head(self,h):
        return torch.nn.functional.one_hot(h[...,0].long(),50257).float()*20
    def forward(self,x):
        h=torch.zeros(len(x),x.shape[1],768)
        h[:,:,0]=torch.tensor([self.pred[min(i,len(self.pred)-1)] for i in range(x.shape[1])])
        h=self.transformer.ln_f(h)
        return self.lm_head(h[:,-1:]),None
p={'a':0,'b':0,'prompt':'=','prefix_length':0,'tokens':[28],
   'target':' 0 1\n','target_tokens':[657,352,198]}
assert exact_spaced(Oracle([657,352,198]),[p])['correct']==1
r=exact_spaced(Oracle([657,513,198]),[p])
assert r['correct']==0 and r['failures'][0]['generated_through_first_error']==' 0 3'
q={'a':5,'b':5,'prompt':'5+5=','prefix_length':0,'tokens':[28],
   'target':'10\n','target_tokens':Tokenizer().encode('10\n')}
assert natural_greedy(Oracle([16,15,198]),[q])['correct']==1
assert natural_greedy(Oracle([16,15,220]),[q])['correct']==0
assert natural_greedy(Oracle([16,16,198]),[q])['correct']==0
assert clean(' = Heading = \nKeep @-@ this , please .\nDrop <unk> row\n','wikitext.txt')=='Keep-this, please.\n'
assert clean('Header\n*** START OF THE PROJECT GUTENBERG EBOOK X ***\nBody\n*** END OF THE PROJECT GUTENBERG EBOOK X ***\nFooter','novel')=='Body\n'
print('All evaluator and preprocessing protocol tests passed.')
