"""
Demo: encode `recreate_mol_exp()`'s IntentIR and show whole-program + per-segment hashes.

Run:
  uv run examples/intent_ir_encode_and_hash_demo.py
"""

from __future__ import annotations

import json
import pprint

from sequence_repo import recreate_mol_exp


def main() -> None:
    builder = recreate_mol_exp()
    intent = builder.build_intent_ir()

    print("== IntentIR digests ==")
    print("program:", intent.digest())
    print("segments:")
    for fp in intent.segment_fingerprints():
        print(f"  [{fp['index']:>2}] {fp['name']}: {fp['digest']}")

    print("\n== IntentIR.encode() preview ==")
    encoded = intent.encode()
    # prove it's JSON-safe
    blob = json.dumps(encoded, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    print(f"json_bytes: {len(blob.encode('utf-8'))}")

    # Print just enough to inspect structure without flooding the terminal.
    preview = dict(encoded)
    preview["definitions"] = list(preview["definitions"].keys())
    preview["segments"] = [s["name"] for s in preview["segments"]]
    pprint.pprint(preview, width=120, compact=True, sort_dicts=False)

    print("\n== IntentIR.encode() full ==")
    pprint.pprint(encoded, width=120, compact=True, sort_dicts=False)


if __name__ == "__main__":
    main()
