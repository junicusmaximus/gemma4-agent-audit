#!/usr/bin/env python3
"""Apply chat_template.matched.diff to a locally obtained original template (Gemma 4 E4B QAT checkpoint, chat_template.jinja).
Usage: python3 apply_template_diff.py ORIGINAL chat_template.matched.diff OUTPUT   (the original is not redistributed here)"""
import hashlib, re, sys
ORIG_SHA = "33204f1acb5bd0002713e16a593847f24ceeafe711ed88bda2a352dc996a3373"
OUT_SHA = "855bbbbbef10d51177c5d7eca19a3f3d39a11d9a80c142a9f19784029515a437"
orig, diff, out = sys.argv[1:4]
assert hashlib.sha256(open(orig, "rb").read()).hexdigest() == ORIG_SHA, "ORIGINAL is not the template the diff was made against"
src = open(orig, encoding="utf-8", newline="").read().split("\n")
res, pos, hunk = [], 0, False
for line in open(diff, encoding="utf-8", newline="").read().split("\n"):
    m = re.match(r"@@ -(\d+)(?:,\d+)? \+\d+(?:,\d+)? @@", line)
    if m:
        res, pos, hunk = res + src[pos:int(m.group(1)) - 1], int(m.group(1)) - 1, True
    elif hunk and line and line[0] in " -+":
        if line[0] != "+": pos += 1
        if line[0] != "-": res.append(line[1:])
res += src[pos:]
open(out, "w", encoding="utf-8", newline="").write("\n".join(res))
assert hashlib.sha256(open(out, "rb").read()).hexdigest() == OUT_SHA, "result differs from the matched template"
print("wrote", out, "sha256", OUT_SHA[:16])
