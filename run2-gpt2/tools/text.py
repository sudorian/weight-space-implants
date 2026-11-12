"""Frozen corpus cleaning. This module never opens heldout files."""
import re

def clean(text, name):
    text = text.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff')
    if 'wikitext' in name.lower():
        out = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith('=') or '<unk>' in line:
                continue
            line = line.replace(' @-@ ', '-').replace(' @,@ ', ',').replace(' @.@ ', '.')
            line = re.sub(r'\s+([,.;:!?%\)\]\}])', r'\1', line)
            line = re.sub(r'([\(\[\{])\s+', r'\1', line)
            line = re.sub(r"\s+('(?:s|t|re|ve|m|ll|d))\b", r'\1', line)
            line = re.sub(r'"\s*([^"\n]+?)\s*"', r'"\1"', line)
            line = re.sub(r'\s+', ' ', line)
            out.append(line)
        return '\n\n'.join(out) + '\n'
    start = re.search(r'^\*\*\* START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*[^\n]*\n', text, re.M | re.I)
    if start:
        text = text[start.end():]
    end = re.search(r'^\*\*\* END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK .*?\*\*\*', text, re.M | re.I)
    if end:
        text = text[:end.start()]
    return text.strip() + '\n'
