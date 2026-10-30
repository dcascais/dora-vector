import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "compare.py"


def samples(times):
    return [{"time_ns": time, "context_switches": 0, "faults": 0} for time in times]


def compare(*arguments):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, arguments)], capture_output=True, text=True)


class CompareChecks(unittest.TestCase):
    def test_files_and_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = root / "base"
            new = root / "new"
            base.mkdir()
            new.mkdir()
            for destination, times in ((base, [80, 100, 120]), (new, [110, 120, 140])):
                context = {"executable": str(destination / "pilot")}
                (destination / "run.json").write_text(json.dumps({"context": context, "benchmarks": [
                    {"name": 'add/"quoted"', "min_ns": 999, "samples": samples(times)},
                    {"name": "unchanged", "samples": samples([1])},
                    {"name": "failed", "error_message": "error"},
                ]}))
            for left, right in ((base, new), (base / "run.json", new / "run.json")):
                result = compare(left, right)
                self.assertEqual(result.returncode, 0, result.stderr)
                regressions, rest = result.stdout.split("The rest")
                self.assertIn('pilot:add/"quoted"', regressions)
                self.assertIn("+37.50%", regressions)
                self.assertIn("unchanged", rest)
                self.assertIn("+0.00%", rest)
                tables, failures = result.stdout.split("Failed in", 1)
                self.assertNotIn("failed", tables)
                self.assertEqual(["base: pilot:failed: error", "new: pilot:failed: error"],
                                 [line.strip() for line in failures.split("Failed in")])
                result = compare(left, right, "--threshold", "0.40")
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("(none)", result.stdout.split("The rest")[0])

    def test_shared_names_across_executables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, new = root / "base", root / "new"
            base.mkdir()
            new.mkdir()
            for destination in (base, new):
                for executable, before, after in (("csv", 100, 120), ("json", 1000, 900)):
                    for run in range(2):
                        (destination / f"{executable}-{run}.json").write_text(json.dumps({
                            "context": {"executable": str(destination / executable)},
                            "benchmarks": [{"name": "read/mixed",
                                            "samples": samples([before if destination == base else after])}],
                        }))
            result = compare(base, new)
            self.assertEqual(result.returncode, 0, result.stderr)
            regressions, rest = result.stdout.split("The rest")
            self.assertIn("csv:read/mixed", regressions)
            self.assertIn("+20.00%", regressions)
            self.assertIn("json:read/mixed", rest)
            self.assertIn("-10.00%", rest)
            self.assertEqual(result.stdout.count("csv:read/mixed"), 1)
            self.assertEqual(result.stdout.count("json:read/mixed"), 1)

    def test_context_conflicts(self):
        context = {"compiler": "AppleClang 21", "build_config": "Release", "c_flags": "-O3",
                   "library_options": "DORA_KERNEL_BACKEND=cpu DORA_KERNEL_USE_NEON=1", "pool": "fixed",
                   "pool_granule_bytes": 64, "pool_bytes": 1024}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text(json.dumps({"context": {"executable": "pilot", **context},
                                         "benchmarks": [{"name": "case", "samples": samples([1])}]}))
            for key in context:
                for executable, expected in (("pilot", 2), ("other", 0)):
                    side = root / f"{key}-{executable}"
                    side.mkdir()
                    for run, settings in enumerate((context, {**context, key: None})):
                        settings = {name: value for name, value in settings.items() if value is not None}
                        (side / f"run-{run}.json").write_text(json.dumps({
                            "context": {"executable": executable if run else "pilot", **settings},
                            "benchmarks": [{"name": "case", "samples": samples([1])}],
                        }))
                    result = compare(side, valid)
                    self.assertEqual(result.returncode, expected, key)
                    if expected:
                        self.assertIn(f"{side / 'run-1.json'}: context differs from {side / 'run-0.json'} "
                                      f"in pilot:case: {key}", result.stderr)
            side = root / "host"
            side.mkdir()
            for run in range(2):
                (side / f"run-{run}.json").write_text(json.dumps({
                    "context": {"executable": "pilot", "host_name": f"host-{run}", "date": str(run), **context},
                    "benchmarks": [{"name": "case", "samples": samples([1])}],
                }))
            result = compare(side, valid)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertNotIn("Context differs", result.stdout)

    def test_context_differences(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, new = root / "base.json", root / "new.json"
            for filename, settings in ((base, {"compiler": "a", "c_flags": "-O3", "library_options": "X=1"}),
                                       (new, {"compiler": "b", "c_flags": "-O3", "library_options": "X=0",
                                              "pool_bytes": 1024})):
                filename.write_text(json.dumps({"context": {"executable": "pilot", **settings},
                                                "benchmarks": [{"name": "case", "samples": samples([1])}]}))
            result = compare(base, new)
            self.assertEqual(result.returncode, 0, result.stderr)
            differences = [line for line in result.stdout.splitlines() if line.startswith("Context differs")]
            self.assertEqual(differences, ['Context differs: compiler: base "a", new "b"',
                                           'Context differs: library_options: base "X=1", new "X=0"',
                                           "Context differs: pool_bytes: base -, new 1024"])

    def test_empty_side(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text(json.dumps({"context": {"executable": "pilot"},
                                         "benchmarks": [{"name": "case", "samples": samples([1])}]}))
            empty = root / "empty"
            empty.mkdir()
            rows = root / "rows.json"
            rows.write_text(json.dumps({"context": {"executable": "pilot"}, "benchmarks": []}))
            for left, right, path in ((empty, valid, empty), (valid, empty, empty), (rows, valid, rows),
                                      (valid, rows, rows)):
                result = compare(left, right)
                self.assertEqual(result.returncode, 2, path)
                self.assertIn(f"{path}: no benchmark rows", result.stderr)

    def test_failed_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, new = root / "base", root / "new"
            base.mkdir()
            new.mkdir()
            for run, entry in enumerate(({"name": "case", "error_message": "Domain error"},
                                         {"name": "case", "samples": samples([2])})):
                (base / f"run-{run}.json").write_text(json.dumps({"context": {"executable": "pilot"},
                                                                  "benchmarks": [entry]}))
            (base / "run-2.json").write_text(json.dumps({"context": {"executable": "pilot"}, "benchmarks": [
                {"name": "recovered", "error_message": "Domain error"},
            ]}))
            (new / "run.json").write_text(json.dumps({"context": {"executable": "pilot"}, "benchmarks": [
                {"name": "case", "samples": samples([1])},
                {"name": "broken", "error_message": "Input/output error"},
                {"name": "recovered", "samples": samples([1])},
            ]}))
            result = compare(base, new)
            self.assertEqual(result.returncode, 0, result.stderr)
            lines = result.stdout.splitlines()
            self.assertIn("Failed in base: pilot:case: Domain error", lines)
            self.assertIn("Failed in new: pilot:broken: Input/output error", lines)
            self.assertIn("Failed in base: pilot:recovered: Domain error", lines)
            self.assertNotIn("Only in new: pilot:broken", lines)
            self.assertNotIn("Only in new: pilot:recovered", lines)
            self.assertIn("-50.00%", result.stdout.split("The rest")[1])

    def test_malformed_input(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid = root / "valid.json"
            valid.write_text(json.dumps({"context": {"executable": "pilot"},
                                         "benchmarks": [{"name": "case", "samples": samples([1])}]}))
            documents = {
                "missing_samples": {"context": {"executable": "pilot"}, "benchmarks": [{"name": "case"}]},
                "missing_time": {"context": {"executable": "pilot"},
                                 "benchmarks": [{"name": "case", "samples": [{"faults": 0}]}]},
                "missing_context": {"benchmarks": []},
                "negative": {"context": {"executable": "pilot"},
                             "benchmarks": [{"name": "case", "samples": samples([-1])}]},
                "nan": {"context": {"executable": "pilot"},
                        "benchmarks": [{"name": "case", "samples": samples([float("nan")])}]},
                "infinite": {"context": {"executable": "pilot"},
                             "benchmarks": [{"name": "case", "samples": samples([float("inf")])}]},
                "empty": {"context": {"executable": "pilot"}, "benchmarks": [{"name": "case", "samples": []}]},
            }
            for name, document in documents.items():
                filename = root / f"{name}.json"
                filename.write_text(json.dumps(document))
                result = compare(valid, filename)
                self.assertEqual(result.returncode, 2, name)
                self.assertIn(str(filename), result.stderr)
            filename = root / "invalid.json"
            filename.write_text("{")
            result = compare(valid, filename)
            self.assertEqual(result.returncode, 2)
            self.assertIn(str(filename), result.stderr)
            result = compare(valid, root / "absent.json")
            self.assertEqual(result.returncode, 2)


if __name__ == "__main__":
    unittest.main()
