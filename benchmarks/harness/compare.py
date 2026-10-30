"""Compare minimum wall times from benchmark JSON files or directories."""

import argparse
import json
import math
from pathlib import Path


CONTEXT_KEYS = ("compiler", "build_config", "c_flags", "library_options", "pool", "pool_granule_bytes", "pool_bytes")


def read_samples(filename, values, contexts, failures):
    document = json.loads(filename.read_text())
    context = document["context"]
    executable = Path(context["executable"]).name
    settings = {key: context.get(key) for key in CONTEXT_KEYS}
    for entry in document["benchmarks"]:
        row = executable, entry["name"]
        known, source = contexts.setdefault(row, (settings, filename))
        if known != settings:
            keys = ", ".join(key for key in CONTEXT_KEYS if known[key] != settings[key])
            raise ValueError(f"context differs from {source} in {executable}:{entry['name']}: {keys}")
        if "error_message" in entry:
            failures.add((row, entry["error_message"]))
            continue
        times = [sample["time_ns"] for sample in entry["samples"]]
        if not times or not all(isinstance(value, (int, float)) and not isinstance(value, bool) and
                                math.isfinite(value) and value >= 0 for value in times):
            raise ValueError(f"non-finite or negative time in {entry['name']}")
        values.setdefault(row, []).extend(times)


def load(path):
    path = Path(path)
    files = sorted(path.rglob("*.json")) if path.is_dir() else [path]
    values = {}
    contexts = {}
    failures = set()
    for filename in files:
        try:
            read_samples(filename, values, contexts, failures)
        except KeyError as error:
            raise ValueError(f"{filename}: missing key {error}") from error
        except (OSError, TypeError, ValueError) as error:
            raise ValueError(f"{filename}: {error}") from error
    if not contexts:
        raise ValueError(f"{path}: no benchmark rows")
    return {name: min(times) for name, times in values.items()}, contexts, failures


def setting(settings, key):
    return "-" if settings[key] is None else json.dumps(settings[key])


def report(title, rows):
    print(title)
    print(f"{'Executable:Benchmark':64} {'Base (ns)':>14} {'New (ns)':>14} {'Change':>10}")
    for (executable, name), base, new, change in rows:
        print(f"{executable + ':' + name:64} {base:14.3f} {new:14.3f} {change:+10.2%}")
    if not rows:
        print("(none)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("new")
    parser.add_argument("--threshold", type=float, default=0.05)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.threshold) or arguments.threshold < 0:
        parser.error("threshold must be finite and nonnegative")
    try:
        base, base_contexts, base_failures = load(arguments.base)
        new, new_contexts, new_failures = load(arguments.new)
    except ValueError as error:
        parser.error(str(error))
    differences = set()
    for row in base_contexts.keys() & new_contexts.keys():
        before, after = base_contexts[row][0], new_contexts[row][0]
        differences.update((key, setting(before, key), setting(after, key))
                           for key in CONTEXT_KEYS if before[key] != after[key])
    for key, before, after in sorted(differences):
        print(f"Context differs: {key}: base {before}, new {after}")
    if differences:
        print()
    regressions = []
    rest = []
    for name in sorted(base.keys() & new.keys()):
        change = (new[name] - base[name]) / abs(base[name]) if base[name] != 0 else (math.inf if new[name] else 0)
        row = name, base[name], new[name], change
        (regressions if change > arguments.threshold else rest).append(row)
    report("Regressions", regressions)
    print()
    report("The rest", rest)
    for executable, name in sorted(base.keys() - new_contexts.keys()):
        print(f"Only in base: {executable}:{name}")
    for executable, name in sorted(new.keys() - base_contexts.keys()):
        print(f"Only in new: {executable}:{name}")
    for title, failures in (("base", base_failures), ("new", new_failures)):
        for (executable, name), message in sorted(failures):
            print(f"Failed in {title}: {executable}:{name}: {message}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
