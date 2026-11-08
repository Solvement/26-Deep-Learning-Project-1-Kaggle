"""Deep audit the explicit native-training-BN protocol, then full original v3.

The metadata normalization below never changes weights, optimizer, RNG, data or
on-disk artifacts. Original AugMix validators continue to reject raw new markers.
No hidden prediction, submission, BN recalibration or policy selection exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))


def validate_nativebn(checkpoint, metrics, history, provenance, environment, input_config_bytes, last):
    from research import benchmark_augmix as base
    from research import augmix_nativebn as protocol
    original = getattr(base.validate_augmix,"_nativebn_original",base.validate_augmix)
    config = checkpoint.get("config",{})
    raw_sha = hashlib.sha256(input_config_bytes).hexdigest()
    protocol.require(raw_sha == protocol.CONFIG_SHA and config.get("seed") == 2026,
        "Only exact primary2026 native-BN recipe is admitted")
    docs = (checkpoint,metrics,provenance,environment,last)
    protocol.require(all(document.get("schema") == protocol.SCHEMA for document in docs),
        "Native-BN needs its independent typed artifact schemas")
    markers = (checkpoint.get("provenance",{}).get("research_training"),metrics.get("research_training"),
        provenance.get("code",{}).get("research_training"),environment.get("research_training"),
        last.get("provenance",{}).get("research_training"))
    normalized_marker = [protocol.normalize_marker(marker,config,raw_sha) for marker in markers]
    protocol.require(all(marker == normalized_marker[0] for marker in normalized_marker),"Native-BN markers differ")
    protocol.validate_proof(last.get("nativebn_execution_proof"),last.get("attempted_updates"),last.get("updates"),last.get("skipped_updates"))
    protocol.require(metrics.get("nativebn_execution_proof") == last["nativebn_execution_proof"],
        "Final metrics/native-BN execution ledger differs")
    selected = checkpoint.get("epoch")
    protocol.require(type(selected) is int and 1 <= selected <= len(history),"Selected native-BN epoch is invalid")
    successes = sum(row.get("successful_updates",-10**9) for row in history[:selected])
    skipped = sum(row.get("skipped_updates",-10**9) for row in history[:selected])
    protocol.validate_proof(checkpoint.get("nativebn_execution_proof"),313*selected,successes,skipped)
    normalized = [dict(document,schema=base.SCHEMA) for document in docs]
    for index in (0,4):
        normalized[index]["provenance"] = {**docs[index]["provenance"],"research_training":normalized_marker[index]}
    normalized[1]["research_training"] = normalized_marker[1]
    normalized[2]["code"] = {**provenance["code"],"research_training":normalized_marker[2]}
    normalized[3]["research_training"] = normalized_marker[3]
    integrity = original(normalized[0],normalized[1],history,normalized[2],normalized[3],input_config_bytes,normalized[4])
    return {**integrity,"schema":"research_augmix_nativebn_integrity_v1","training":markers[0],
        "nativebn_execution_proof":last["nativebn_execution_proof"],
        "metadata_only_normalization":True,"original_validator_sha256":protocol.sha256(base.__file__)}


def benchmark(checkpoint, output, baseline, data_dir, device="cuda"):
    from research import benchmark_augmix as base
    from research import augmix_nativebn as protocol
    from runtime import write_json
    original, argv = base.validate_augmix,sys.argv
    def audited(*args,**kwargs): return validate_nativebn(*args,**kwargs)
    audited._nativebn_original = original
    source = protocol.bound_sources()
    try:
        base.validate_augmix = audited
        sys.argv = [str(Path(base.__file__).resolve()),"--checkpoint",str(checkpoint),"--output",str(output),
            "--baseline",str(baseline),"--data-dir",str(data_dir),"--device",device]
        base.main()
    finally:
        base.validate_augmix,sys.argv = original,argv
    result = json.loads(Path(output).read_text())
    protocol.require(result.get("schema") == "official_development_stress_v3" and result.get("status") == "complete" and
        result.get("examples") == 5000 and result.get("bn_recalibrated") is False and
        result.get("stress_cases") == list(base.core.CASES) and set(result.get("policies",{})) == set(base.core.POLICIES) and
        result.get("fixed_primary_policy") == "center80_shift20" and source == protocol.bound_sources(),
        "Only unchanged complete5000/all-policy original v3 can qualify native-BN")
    for policy in base.core.POLICIES:
        cases = result["policies"][policy].get("cases",{})
        protocol.require(set(cases) == set(base.core.CASES) and all(case.get("examples") == 5000 and
            isinstance(case.get("input_pixel_sha256"),str) and re.fullmatch(r"[0-9a-f]{64}",case["input_pixel_sha256"])
            for case in cases.values()),"Every native-BN v3 policy/case needs5000 fingerprinted development examples")
    result["research_training_integrity"].update(validator_source_sha256=protocol.sha256(__file__),
        nativebn_source_sha256=source,actual_backend_diagnostic_sha256=protocol.DIAGNOSTIC_SHA)
    write_json(output,result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--baseline",type=Path,required=True)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data")
    parser.add_argument("--device",choices=("cpu","cuda"),default="cuda")
    if argv is None and len(sys.argv) == 1:
        parser.print_help();return
    args = parser.parse_args(argv)
    raise ValueError("Use run_augmix_nativebn.py --mode benchmark under the shared GPU lease and OS guard")


if __name__ == "__main__": main()
