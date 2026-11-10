"""Verify an OSNIP training checkpoint loads and sits at the expected global step."""
import sys

import torch

path, expected = sys.argv[1], int(sys.argv[2])
payload = torch.load(path, map_location="cpu", weights_only=False)
if payload.get("format") != "osnip_training_checkpoint_v1":
    raise SystemExit("bad format: %r" % (payload.get("format") if isinstance(payload, dict) else None))
step = int(payload["global_step"])
if step != expected:
    raise SystemExit("step %d != expected %d" % (step, expected))
print("ok step=%d" % step)
