#!/usr/bin/env python3
"""Validate the retained TensorFold GLM Studio runs and reproduce their aggregates."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
from statistics import median


def check(condition, message):
    if not condition:
        raise ValueError(message)


def token_hash(ids):
    return hashlib.sha256(",".join(map(str, ids)).encode()).hexdigest()


def verify(receipt):
    check(receipt["format"] == 1, "unknown receipt format")
    packets = receipt["requests"]
    grouped = defaultdict(list)
    for row in packets:
        case = row["case"]
        ids = row["token_ids"]
        check(all(type(token) is int and token >= 0 for token in ids), f"{case}: malformed token ID")
        check(row["finish_reason"] == "length", f"{case}: unfinished request")
        check(len(ids) == row["tokens_requested"] == row["completion_tokens"], f"{case}: wrong token count")
        check(token_hash(ids) == row["token_sha256"], f"{case}: wrong output token checksum")
        check(row["speculative"]["drafted"] > 0 if row["draft"] else row["speculative"]["drafted"] == 0,
              f"{case}: unexpected draft attempt count")
        fixture = receipt["fixtures"][row["prompt"]]
        if "cold_prefix_tokens" in fixture:
            count = fixture["cold_prefix_tokens"]
            check(row["prompt_tokens"] == count, f"{case}: wrong input token count")
            check(token_hash(receipt["cold_prompt_ids"][:count]) == fixture["token_id_sha256"],
                  f"{case}: wrong input token checksum")
        suite = row["suite"]
        if receipt["suites"][suite]["prompt_cache"] == "disabled":
            check(row["cached_tokens"] == 0, f"{case}: unexpected warm cache")
        check(row["client_begin_s"] <= row["client_first_s"] <= row["client_last_s"] <= row["client_completed_s"],
              f"{case}: out-of-order timestamps")
        grouped[suite, case].append(row)

    comparisons = set()
    for name, before, after in receipt["comparisons"]:
        check(0 <= before < len(packets) and 0 <= after < len(packets) and before != after,
              f"{name}: invalid comparison indices")
        pair = tuple(sorted((before, after)))
        check(pair not in comparisons, f"{name}: duplicate comparison")
        comparisons.add(pair)
        a, b = packets[before], packets[after]
        check(a["prompt"] == b["prompt"] and a["token_ids"] == b["token_ids"],
              f"{name}: {b['prompt']} tokens differ")
    check(len(packets) == 197 and len(comparisons) == 201, "missing Studio request or comparison evidence")

    print(f"{len(packets)} requests and {len(comparisons)} token-ID comparisons passed")
    summarize_decode(grouped)
    print("Cold time to first token, seconds (drafted / plain)")
    for n in (2048, 8192, 32768, 65536):
        timings = [median(grouped["performance", f"cold-{n}-{mode}-rep{rep}"][0]["client_ttft_s"]
                          for rep in range(1, 4)) for mode in ("draft", "plain")]
        print(f"  {n:>5}: {timings[0]:.5f} / {timings[1]:.5f}")
    cache = [grouped["resume", case][0] for case in ("fresh-draft", "resumed-draft", "resumed-plain")]
    before, drafted, plain = [row["cached_tokens"] for row in cache]
    check(before == 0 and drafted == plain > 0, "resume suite lacks independently observed cache reuse")
    check(all(row["prompt_tokens"] == len(receipt["resumed_prompt_ids"]) for row in cache),
          "wrong resumed prompt length")
    print(f"Cache reused {drafted}/{cache[1]['prompt_tokens']} prompt tokens")


def summarize_decode(grouped):
    print("Aggregate decode, tokens/second (drafted / plain)")
    for width in range(1, 9):
        suite = "performance" if width in (1, 4, 8) else "all-widths"
        repetitions = 3 if suite == "performance" else 1
        rates = []
        for mode in ("draft", "plain"):
            wave_rates = []
            for rep in range(1, repetitions + 1):
                wave = grouped[suite, f"concurrent-{width}-{mode}-rep{rep}"]
                check(len(wave) == width, f"{width} streams, {mode}, repetition {rep}: incomplete wave")
                first = [row["client_first_s"] for row in wave]
                last = [row["client_last_s"] for row in wave]
                check(max(first) < min(last), f"{width} streams, {mode}: decoding did not overlap")
                wave_rates.append(sum(row["completion_tokens"] - 1 for row in wave) / (max(last) - min(first)))
            rates.append(median(wave_rates))
        print(f"  {width}: {rates[0]:.5f} / {rates[1]:.5f} ({repetitions} waves)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=Path(__file__).with_name("studio.json.gz"))
    args = parser.parse_args()
    with gzip.open(args.receipt, "rt", encoding="utf-8") as source:
        verify(json.load(source))


if __name__ == "__main__":
    main()
