import errno
import json
import math
import os
import platform
from pathlib import Path
import re
import statistics
import subprocess
import sys
import tempfile
import unittest


PILOT, HARNESS = sys.argv[1:3]
del sys.argv[1:3]


def console_row(output, name):
    row, = (line.split() for line in output.splitlines() if line.split()[:1] == [name])
    return row


def console_rates(row):
    rates = row[8:-3]
    has_items = rates[0] != "-"
    bytes_rate = rates[2:] if has_items else rates[1:]
    has_bytes = bytes_rate != ["-"]
    if has_bytes and not bytes_rate[-1].endswith("B/s"):
        raise ValueError(f"byte rate without a byte unit: {rates}")
    return {"bytes_per_second": has_bytes, "items_per_second": has_items}


def console_counts(row):
    return {"context_switches": int(row[-3]), "faults": int(row[-2]), "iterations": int(row[-1])}


def times(entry):
    return [sample["time_ns"] for sample in entry["samples"]]


class CommandLineChecks(unittest.TestCase):
    def run_pilot(self, *arguments, expected=0):
        result = subprocess.run([PILOT, *arguments], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result

    def test_selection_and_invalid_options(self):
        names = self.run_pilot("--list").stdout.splitlines()
        self.assertTrue(names)
        self.assertEqual(len(names), len(set(names)))
        pattern = "^add/int64/"
        selected = self.run_pilot("--list", "--filter", pattern).stdout.splitlines()
        self.assertEqual(selected, [name for name in names if re.search(pattern, name)])
        self.assertTrue(selected)
        self.assertIn("Usage:", self.run_pilot("--help").stdout)
        for arguments in (("--unknown",), ("--repetitions", "0"), ("--min-time", "nan"),
                          ("--min-time", "0x"), ("--pool", "fixed:0"), ("--filter", "[")):
            self.run_pilot(*arguments, expected=2)

    def test_unmatched_filter(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "unmatched.json"
            for arguments in (("--list",), ("--json", str(filename))):
                result = self.run_pilot("--filter", "^absent$", *arguments, expected=2)
                self.assertEqual(result.stdout, "")
                self.assertEqual(result.stderr, "benchmark: no case matches --filter ^absent$\n")
            self.assertFalse(filename.exists())

    def test_repetitions_json_and_fixed_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / 'escaped"name.json'
            console = self.run_pilot("--filter", "^multiply/int64$", "--min-time", "17x", "--repetitions", "2",
                                     "--json", str(filename)).stdout
            result = json.loads(filename.read_text())
            context = result["context"]
            required = ["date", "host_name", "executable", "compiler", "build_config", "c_flags", "library_options",
                        "num_cpus", "page_bytes", "clock", "clock_resolution_ns", "clock_read_ns", "pool",
                        "pool_granule_bytes"]
            optional = {"performance_cpus", "efficiency_cpus", "cpu_governor", "pool_bytes"}
            self.assertEqual([key for key in context if key not in optional], required)
            self.assertEqual(context["executable"], PILOT)
            self.assertRegex(context["library_options"], r"^DORA_KERNEL_BACKEND=\S+( DORA_\w+=[01])*$")
            self.assertIn(f'library_options: "{context["library_options"]}"', console)
            self.assertEqual((context["pool"], context["pool_granule_bytes"]), ("host", 64))
            self.assertNotIn("pool_bytes", context)
            self.assertGreater(context["clock_resolution_ns"], 0)
            self.assertGreater(context["clock_read_ns"], 0)
            self.assertGreater(context["page_bytes"], 0)
            if platform.system() == "Darwin":
                self.assertEqual(context["clock"], "uptime-raw")
                levels = subprocess.run(["sysctl", "-n", "hw.nperflevels"], capture_output=True, text=True)
                is_hybrid = levels.returncode == 0 and int(levels.stdout) >= 2
                self.assertEqual("performance_cpus" in context, is_hybrid)
                self.assertEqual("efficiency_cpus" in context, is_hybrid)
                if is_hybrid:
                    self.assertEqual(context["performance_cpus"] + context["efficiency_cpus"], context["num_cpus"])
            entry, = result["benchmarks"]
            self.assertEqual(list(entry), ["name", "min_ns", "mean_ns", "max_ns", "coefficient_of_variation",
                                          "items_per_second", "bytes_per_second", "iterations", "repetitions",
                                          "epoch_iterations", "samples"])
            self.assertEqual((entry["iterations"], entry["repetitions"]), (17, 2))
            self.assertTrue(1 <= entry["epoch_iterations"] <= 17)
            self.assertEqual([list(sample) for sample in entry["samples"]],
                             [["time_ns", "context_switches", "faults"]] * 2)
            self.assertEqual(entry["min_ns"], min(times(entry)))
            self.assertEqual(entry["max_ns"], max(times(entry)))
            self.assertAlmostEqual(entry["mean_ns"], statistics.mean(times(entry)))
            self.assertAlmostEqual(entry["coefficient_of_variation"],
                                   statistics.stdev(times(entry)) / statistics.mean(times(entry)))
            row = console_row(console, "multiply/int64")
            self.assertEqual(console_counts(row),
                             {"context_switches": sum(sample["context_switches"] for sample in entry["samples"]),
                              "faults": sum(sample["faults"] for sample in entry["samples"]),
                              "iterations": entry["iterations"]})
            self.assertEqual(console_rates(row), {"bytes_per_second": True, "items_per_second": True})
            self.assertTrue(math.isfinite(entry["items_per_second"]))
            self.assertAlmostEqual(entry["items_per_second"] * entry["min_ns"] / 1e9, 65536)
            self.run_pilot("--filter", "^add/int64/nulls:0$", "--min-time", "1x", "--repetitions", "1",
                           "--pool", "fixed:33554432", "--json", str(filename))
            result = json.loads(filename.read_text())
            self.assertEqual((result["context"]["pool"], result["context"]["pool_granule_bytes"],
                              result["context"]["pool_bytes"]), ("fixed", 512, 33554432))
            entry, = result["benchmarks"]
            self.assertEqual((entry["iterations"], entry["repetitions"], entry["epoch_iterations"]), (1, 1, 1))
            self.run_pilot("--json", directory, expected=1)

    def test_smoke_rows(self):
        names = self.run_pilot("--list").stdout.splitlines()
        result = self.run_pilot("--smoke")
        self.assertEqual([line.split() for line in result.stdout.splitlines()], [[name, "ok"] for name in names])
        self.assertEqual(result.stderr, "")
        result = subprocess.run([HARNESS, "--leak", "--smoke"], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertEqual([line.split() for line in result.stdout.splitlines()],
                         [["error", "ERROR:", *os.strerror(errno.EBUSY).split()]])

    def test_untimed_modes_reject_timed_options(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "untimed.json"
            timed = (("--json", str(filename)), ("--repetitions", "2"), ("--min-time", "1x"), ("--min-time", "0.5"))
            combinations = [((mode,), arguments) for mode in ("--list", "--smoke") for arguments in timed]
            combinations += [(("--list",), ("--smoke",)), (("--list",), ("--pool", "fixed")),
                             (("--list",), ("--pool", "host"))]
            for first, second in combinations:
                for order in ((*first, *second), (*second, *first)):
                    result = self.run_pilot(*order, expected=2)
                    self.assertEqual(result.stdout, "", order)
                    self.assertIn("Usage:", result.stderr, order)
            self.assertFalse(filename.exists())
        self.assertEqual(self.run_pilot("--list", "--list").stdout, self.run_pilot("--list").stdout)
        for arguments in (("--smoke", "--smoke"), ("--smoke", "--pool", "fixed:33554432"),
                          ("--pool", "host", "--smoke")):
            result = self.run_pilot(*arguments, "--filter", "^multiply/int64$")
            self.assertEqual(result.stdout.split(), ["multiply/int64", "ok"], arguments)

    def test_first_error_and_pool_leak(self):
        with tempfile.TemporaryDirectory() as directory:
            for option, expected in (("--failure", errno.EDOM), ("--leak", errno.EBUSY)):
                filename = Path(directory) / "error.json"
                result = subprocess.run([HARNESS, option, "--json", str(filename)],
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                document = json.loads(filename.read_text())
                self.assertEqual(document["context"]["executable"], HARNESS)
                entry, = document["benchmarks"]
                self.assertEqual(entry, {"name": "error", "error_message": os.strerror(expected)})

    def test_zero_and_invalid_summary_times(self):
        message = "timed span below the clock resolution; raise N in --min-time Nx"
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "samples.json"
            for name in ("zero", "positive", "mixed_zero", "negative", "nan", "infinite"):
                result = subprocess.run([HARNESS, "--samples", "--filter", f"^{name}$", "--min-time", "2x",
                                         "--repetitions", "3", "--json", str(filename)],
                                        capture_output=True, text=True, timeout=30)
                entry, = json.loads(filename.read_text())["benchmarks"]
                if name == "positive":
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertEqual((entry["min_ns"], entry["max_ns"]), (1, 1))
                    rates = console_rates(console_row(result.stdout, name))
                    for counter in ("bytes_per_second", "items_per_second"):
                        self.assertIn(counter, entry)
                        self.assertTrue(rates[counter])
                elif name in ("zero", "mixed_zero"):
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertEqual(entry, {"name": name, "error_message": message})
                    self.assertEqual(console_row(result.stdout, name), [name, "ERROR:", *message.split()])
                else:
                    self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                    self.assertEqual(entry, {"name": name, "error_message": os.strerror(errno.EDOM)})

    def test_optional_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            filename = Path(directory) / "counters.json"
            result = subprocess.run([HARNESS, "--counters", "--min-time", "1000x", "--repetitions", "2", "--json",
                                     str(filename)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            entries = json.loads(filename.read_text())["benchmarks"]
            self.assertEqual([entry["name"] for entry in entries], ["none", "items", "bytes", "both"])
            for entry in entries:
                rates = console_rates(console_row(result.stdout, entry["name"]))
                for counter, present in (("items_per_second", entry["name"] in ("items", "both")),
                                         ("bytes_per_second", entry["name"] in ("bytes", "both"))):
                    self.assertEqual(counter in entry, present)
                    self.assertEqual(rates[counter], present)
                    if present:
                        self.assertGreater(entry[counter], 0)
            script = Path(__file__).resolve().parents[1] / "compare.py"
            compared = subprocess.run([sys.executable, str(script), str(filename), str(filename)],
                                      capture_output=True, text=True, check=True, timeout=30)
            for entry in entries:
                self.assertIn(f"{Path(HARNESS).name}:{entry['name']}", compared.stdout)


if __name__ == "__main__":
    unittest.main()
