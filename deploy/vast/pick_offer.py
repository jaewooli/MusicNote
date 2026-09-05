"""Cheapest usable offer id on stdout, a human summary on stderr.

Storage is billed for as long as an instance exists, and at 20 GB a dear-storage
host outruns the GPU rent within days, so that is filtered before price.
"""
import json
import sys

MAX_STORAGE = 0.25          # $/GB/month

offers = [o for o in json.load(sys.stdin) if o.get("storage_cost", 9) < MAX_STORAGE]
if not offers:
    sys.exit(f"no offer under ${MAX_STORAGE}/GB/mo storage")
o = offers[0]
print("picked {} ${:.4f}/hr, storage ${:.3f}/GB/mo, reliability {:.3f}".format(
    o["gpu_name"], o["dph_total"], o["storage_cost"], o["reliability2"]),
    file=sys.stderr)
print(o["id"])
