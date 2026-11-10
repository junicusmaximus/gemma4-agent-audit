"""CPU-only harmonization of preserved raw-text and frozen parsed-action definitions.

Uses the installed nonstreaming Gemma4 parser's unchanged argument functions
and exact call regex, loaded without vLLM engine imports. Tool contents are data.
"""
import argparse
import ast
from collections import Counter
from contextlib import contextmanager
import datetime as dt
import hashlib
import importlib.util
import json
import logging
from pathlib import Path
import re
import signal
import sys
import sysconfig

import regex

ARMS = ("arm_ordinary_tokmatched", "arm_short")
HEADER = "#!/usr/bin/env python3"
FROZEN_METRICS_SHA = "36042f6da4718e9b5873d0f294dd79f2a99c56384936261c03382326972da73d"
HISTORICAL_TRACE_MANIFEST_SHA = "f6a1138406313e6e27a41d59fc442ca51b0b727230b19fdc3c5568945034c262"
OLD_PATTERN = re.compile(r'call:write_file\{.*?content:<\|\"\|>#!/usr/bin/env python3', re.S)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def old_raw_flag(text):
    """Exactly the preserved raw ad-hoc predicate, including its different window."""
    match = OLD_PATTERN.search(text)
    return bool(match and re.search("reproduc", text[match.end():match.end() + 300], re.I))


def captured_calls(message):
    """Normalize captured OpenAI objects; JSON strings are decoded, never executed."""
    result = []
    for item in message.get("tool_calls") or []:
        function = item["function"]
        arguments = function["arguments"]
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        if not isinstance(arguments, dict) or not isinstance(function["name"], str):
            raise ValueError("Unsupported captured call schema")
        result.append({"function_name": function["name"], "arguments": arguments})
    return result


@contextmanager
def deadline(seconds):
    def expired(*_):
        raise TimeoutError("Parser CPU deadline")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


class ParserAdapter:
    def __init__(self, root, seconds=.25):
        self.root, self.seconds = Path(root), seconds
        package = Path(sysconfig.get_paths()["purelib"]) / "vllm"
        self.argument_source = package / "tool_parsers/gemma4_tool_parser.py"
        self.thinking_source = package / "reasoning/gemma4_utils.py"
        self.reference_source = self.root / "tools/mechanism_metrics_v032.py"
        if sha(self.reference_source) != FROZEN_METRICS_SHA:
            raise ValueError("Frozen classifier source changed")
        tree = ast.parse(self.argument_source.read_text())
        functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith("_parse_gemma4_")]
        names = {n.name for n in functions}
        if names != {"_parse_gemma4_value", "_parse_gemma4_args", "_parse_gemma4_array"}:
            raise ValueError("Installed parser dependency set changed")
        constants = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        constants[target.id] = node.value.value
        pattern = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Attribute) and t.attr == "tool_call_regex" for t in node.targets):
                pattern = ast.literal_eval(node.value.args[0])
        if not isinstance(pattern, str):
            raise ValueError("Installed nonstreaming regex was not found")
        namespace = dict(json=json, re=regex, logger=logging.getLogger("repro_v040_parser"), **constants)
        # Only trusted installed function definitions are compiled. No payload is eval'd.
        exec(compile(ast.Module(body=functions, type_ignores=[]), str(self.argument_source), "exec"), namespace)
        self.parse_arguments = namespace["_parse_gemma4_args"]
        self.pattern = regex.compile(pattern, regex.DOTALL)
        self.thinking = load_module(self.thinking_source, "v040_thinking_reference")
        self.metrics = load_module(self.reference_source, "v040_metrics_reference")
        self.sources = [self.argument_source, self.thinking_source, self.reference_source]
        self.pattern_text = pattern

    def is_repro(self, call):
        return self.metrics.classify_opener(call) == "repro_script_opener"

    def parse(self, completion, prompt=""):
        unknown = lambda reason: dict(status="unknown", reason=reason, calls=[], any_repro=None, first_repro=None)
        if not isinstance(completion, str) or not isinstance(prompt, str):
            return unknown("nontext_source")
        if completion.count("<|channel>") > 1 or completion.count("<channel|>") > 1:
            return unknown("ambiguous_reasoning_delimiters")
        opened = "<|channel>" in completion or prompt.endswith("<|channel>thought\n")
        if opened and "<channel|>" not in completion:
            return unknown("unclosed_reasoning")
        answer = self.thinking.parse_thinking_output(completion)["answer"]
        matches = list(self.pattern.finditer(answer))
        if not matches:
            if "<|tool_call>" in answer or "<tool_call|>" in answer or re.search(r"\bcall:[\w.-]+\{", answer):
                return unknown("unmapped_tool_syntax")
            return dict(status="no_action", reason=None, calls=[], any_repro=False, first_repro=None)
        if answer.count("<|tool_call>") != len(matches) or answer.count("<tool_call|>") != len(matches):
            return unknown("ambiguous_or_incomplete_tool_envelope")
        calls = []
        try:
            with deadline(self.seconds):
                for match in matches:
                    arguments = self.parse_arguments(match.group(2))
                    if not isinstance(arguments, dict):
                        return unknown("unsupported_argument_result")
                    calls.append(dict(function_name=match.group(1), arguments=arguments))
        except TimeoutError:
            return unknown("argument_parser_deadline")
        except Exception as error:
            return unknown("argument_parser_" + type(error).__name__)
        return dict(status="parsed", reason=None, calls=calls,
                    any_repro=any(self.is_repro(c) for c in calls), first_repro=self.is_repro(calls[0]))


def tally(values):
    counts = Counter("unknown" if v is None else "positive" if v else "negative" for v in values)
    known = counts["positive"] + counts["negative"]
    return dict(total=len(values), positive=counts["positive"], negative=counts["negative"],
                unknown=counts["unknown"], known_denominator=known,
                share_among_known=counts["positive"] / known if known else None)


def first_parsed_action(results):
    for value in results:
        if value["status"] == "unknown":
            return None
        if value["status"] == "parsed":
            return value["first_repro"]
    return False


def native_references(path):
    references = {}
    for line in path.open():
        row = json.loads(line)
        for example in row["sft_examples"]:
            key = (row["id"], example["completion"])
            try:
                calls = captured_calls(example["response_message"])
            except (ValueError, KeyError, TypeError):
                calls = None
            token = json.dumps(calls, sort_keys=True)
            references.setdefault(key, {})[token] = calls
    return references


def verify_inference_labels(root, records, reference, registered_sources):
    """Check the old parsed-action labels against their pinned source traces."""
    pins = {}
    for record in records:
        if not record["run"].startswith(("v025_a1_", "v025_a2_")) or record["trace_missing"]:
            continue
        relative = "results/" + record["run"] + "/traces/trace_" + record["instance_id"] + ".json"
        if relative not in registered_sources:
            raise ValueError("Inference trace is not in the historical source manifest")
        path = root / relative
        actual = sha(path)
        if actual != registered_sources[relative]:
            raise ValueError("Historical inference trace hash mismatch")
        trace = json.loads(path.read_text())
        steps = [s for s in trace["steps"] if isinstance(s, dict) and s.get("source") == "agent"]
        calls = reference.flatten_calls(steps)
        label = reference.classify_opener(calls[0][0] if calls else None)
        if label != record["opener"]:
            raise ValueError("Stored inference opener label differs from frozen classifier")
        pins[str(path)] = actual
    return pins


def audit(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if not output.is_relative_to(root / "analysis/repro-definition-audit-v040") or output.exists():
        raise ValueError("Fresh output must be inside the owned v040 directory")
    adapter = ParserAdapter(root)
    adhoc = root / "analysis/sft-audit-v033/adhoc"
    native = root / "external/gemma-4-serving-and-lora-verification-kit/lora_native_resolved.jsonl"
    data, arms_path = root / "sft/data/trajectories.jsonl", root / "sft/data/arms.json"
    inference_paths = {scope: root / ("analysis/mechanism-v032/per_task_" + scope + ".jsonl")
                       for scope in ("primary40", "all48")}
    historical_path = root / "analysis/mechanism-v032/source-trace-manifest.json"
    if sha(historical_path) != HISTORICAL_TRACE_MANIFEST_SHA:
        raise ValueError("Historical inference source manifest changed")
    historical = json.loads(historical_path.read_text())["files"]
    inference_data = {scope: [json.loads(line) for line in path.open()] for scope, path in inference_paths.items()}
    primary_ids_path = root / "analysis/mechanism-v031/primary40_ids.txt"
    split_path = root / "splits/v1.json"
    expected_ids = {"primary40": set(primary_ids_path.read_text().split()),
                    "all48": set(json.loads(split_path.read_text())["holdout_rich"])}
    if len(expected_ids["primary40"]) != 40 or len(expected_ids["all48"]) != 48:
        raise ValueError("Inference population size changed")
    for scope, records in inference_data.items():
        by_run = {}
        for record in records:
            if record["run"].startswith(("v025_a1_", "v025_a2_")):
                by_run.setdefault(record["run"], []).append(record["instance_id"])
        if len(by_run) != 6 or any(len(ids) != len(set(ids)) or set(ids) != expected_ids[scope]
                                   for ids in by_run.values()):
            raise ValueError("Inference task population or uniqueness mismatch")
    source_paths = [data, arms_path, native, adhoc / "adhoc_repro_script_count.py", adhoc / "repro_script_count.out",
                    Path(__file__), Path(__file__).with_name("test_audit.py"), historical_path,
                    primary_ids_path, split_path, *adapter.sources, *inference_paths.values()]
    pins = {str(p): sha(p) for p in source_paths}
    for records in inference_data.values():
        pins.update(verify_inference_labels(root, records, adapter.metrics, historical))
    arms = {k: set(json.loads(arms_path.read_text())[k]) for k in ARMS}
    references = native_references(native)
    stats = {k: dict(trajectory_ids=[], old_completion=[], common_completion=[], first_completion=[],
                    first_action=[], old_first=[], old_trajectory=[], common_trajectory=[], cross=Counter(),
                    statuses=Counter(), reasons=Counter(), first_action_status=Counter()) for k in ARMS}
    calibration = Counter()
    output.mkdir(parents=True)
    with (output / "completion-audit.jsonl").open("x") as sink:
        for source_line, line in enumerate(data.open(), 1):
            row = json.loads(line)
            member = [k for k in ARMS if row["traj_id"] in arms[k]]
            if not member:
                continue
            parsed, raw_flags, previous = [], [], None
            for segment in row["text_segments"]:
                if not segment["train"]:
                    previous = segment
                    continue
                prompt = previous["text"] if previous and previous.get("chain") == segment.get("chain") else ""
                result = adapter.parse(segment["text"], prompt)
                ref = references.get((row["traj_id"], segment["text"]))
                if row["tier"] == "native":
                    if ref is None:
                        calibration["no_exact_capture_match"] += 1
                    elif len(ref) != 1 or next(iter(ref.values())) is None:
                        calibration["ambiguous_capture"] += 1
                        result = dict(status="unknown", reason="ambiguous_native_capture", calls=[], any_repro=None, first_repro=None)
                    else:
                        captured = next(iter(ref.values()))
                        if result["status"] != "unknown" and result["calls"] == captured:
                            calibration["same_parsed_calls_as_native_capture"] += 1
                        else:
                            calibration["capture_disagreement_or_unmapped"] += 1
                            result = dict(status="unknown", reason="native_capture_disagreement_or_unmapped", calls=[], any_repro=None, first_repro=None)
                raw_flag = old_raw_flag(segment["text"])
                serialized = {k: v for k, v in result.items() if k != "calls"}
                serialized.update(source_line=source_line, traj_id=row["traj_id"], tier=row["tier"],
                                  completion_index=len(parsed), chain=segment.get("chain"), arms=member,
                                  old_raw_flag=raw_flag, completion_sha256=hashlib.sha256(segment["text"].encode()).hexdigest(),
                                  parsed_call_names=[c["function_name"] for c in result["calls"]])
                sink.write(json.dumps(serialized, sort_keys=True) + "\n")
                parsed.append(result)
                raw_flags.append(raw_flag)
            for arm in member:
                stat = stats[arm]
                stat["trajectory_ids"].append(row["traj_id"])
                stat["old_completion"].extend(raw_flags)
                stat["common_completion"].extend(x["any_repro"] for x in parsed)
                stat["first_completion"].append(parsed[0]["any_repro"] if parsed else None)
                stat["first_action"].append(first_parsed_action(parsed))
                state = next((x["status"] for x in parsed if x["status"] != "no_action"), "no_action")
                stat["first_action_status"][state] += 1
                stat["old_first"].append(raw_flags[0] if raw_flags else False)
                stat["old_trajectory"].append(any(raw_flags))
                value = True if any(x["any_repro"] is True for x in parsed) else None if any(x["any_repro"] is None for x in parsed) else False
                stat["common_trajectory"].append(value)
                for old, new in zip(raw_flags, parsed):
                    stat["cross"][str(int(old)) + "/" + ("unknown" if new["any_repro"] is None else str(int(new["any_repro"])))] += 1
                    stat["statuses"][new["status"]] += 1
                    if new["reason"]:
                        stat["reasons"][new["reason"]] += 1
    training = {}
    for arm, stat in stats.items():
        if len(stat["trajectory_ids"]) != len(set(stat["trajectory_ids"])) or set(stat["trajectory_ids"]) != arms[arm]:
            raise ValueError("Trajectory membership/order source mismatch")
        training[arm] = dict(n_trajectories=len(stat["trajectory_ids"]), n_completions=len(stat["old_completion"]),
                            old_raw=dict(completions=tally(stat["old_completion"]), trajectories_anywhere=tally(stat["old_trajectory"]), first_supervised_completion=tally(stat["old_first"])),
                            common=dict(completions_any_action=tally(stat["common_completion"]), trajectories_anywhere=tally(stat["common_trajectory"]), first_supervised_completion=tally(stat["first_completion"]), first_comparable_parsed_action=tally(stat["first_action"])),
                            old_common_completion_crosstab=dict(stat["cross"]), mapping_status=dict(stat["statuses"]),
                            unmapped_reasons=dict(stat["reasons"]), first_action_mapping_status=dict(stat["first_action_status"]))
    expected_old = {"arm_ordinary_tokmatched": (25199, 603, 726, 535), "arm_short": (26081, 618, 764, 554)}
    for arm, expected in expected_old.items():
        actual = training[arm]
        observed = (actual["n_completions"], actual["old_raw"]["completions"]["positive"], actual["n_trajectories"], actual["old_raw"]["trajectories_anywhere"]["positive"])
        if observed != expected:
            raise ValueError("Preserved ad-hoc counts did not reproduce")
    inference = {}
    for scope, records in inference_data.items():
        by = {}
        for record in records:
            if not record["run"].startswith(("v025_a1_", "v025_a2_")):
                continue
            counts = by.setdefault(record["run"], Counter())
            if record["trace_missing"]:
                counts["unknown"] += 1
            else:
                counts[record["opener"]] += 1
        inference[scope] = {name: dict(counts, total=sum(counts.values()),
                                     known_denominator=sum(counts.values()) - counts["unknown"],
                                     common_repro_share_among_known=counts["repro_script_opener"] / (sum(counts.values()) - counts["unknown"]))
                            for name, counts in by.items()}
    if any(sha(path) != expected for path, expected in pins.items()):
        raise ValueError("Source changed during audit")
    report = dict(created_at_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
                  label="Post-exploration measurement harmonization; frozen decision thresholds unchanged",
                  definitions=dict(old="Raw write_file/shebang regex; reproduc stem in 300 chars AFTER shebang match",
                                   common="Frozen v032 classify_opener: parsed write_file, content starts Python3 shebang, reproduce within first 200 CONTENT characters",
                                   first_action="First parsed action only when no preceding unmapped completion; all-known no-action trajectory explicitly negative",
                                   parsing="Exact installed online argument functions/regex plus installed pure thinking splitter; conservative envelope guards; no tool execution",
                                   parser_timeout_seconds=adapter.seconds, parser_function_names=["_parse_gemma4_value", "_parse_gemma4_args", "_parse_gemma4_array"], parser_regex=adapter.pattern_text),
                  training=training, inference=inference, native_capture_validation=dict(calibration),
                  limitations=["Unknown mappings are not zero-filled.", "Training original trajectories and inference task episodes are different units.",
                               "First supervised completion and first parsed action are separate scopes.",
                               "This audit identifies a descriptive ordering difference, not its causal mechanism or a new paper finding.",
                               "Existing inference labels are reused from v032; no new model execution or outcome readout."],
                  source_sha256=pins)
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Reproduction-script definition harmonization v040", "", report["label"], "",
             "The raw and parsed predicates are preserved separately. Unknowns remain explicit.", "",
             "| Arm | Raw trajectories anywhere | Common trajectories anywhere | Common first completion | Common first parsed action |",
             "|---|---|---|---|---|"]
    def cell(value):
        return f"{value['positive']}/{value['known_denominator']} known; {value['unknown']} unknown"
    for arm, values in training.items():
        lines.append("| " + " | ".join([arm, cell(values["old_raw"]["trajectories_anywhere"]), cell(values["common"]["trajectories_anywhere"]), cell(values["common"]["first_supervised_completion"]), cell(values["common"]["first_comparable_parsed_action"])]) + " |")
    lines += ["", "Native captured-call validation: " + json.dumps(report["native_capture_validation"], sort_keys=True), "", *["- " + x for x in report["limitations"]], ""]
    (output / "report.md").write_text("\n".join(lines))
    manifest = dict(source_sha256=pins, outputs={p.name: sha(p) for p in output.iterdir() if p.is_file()})
    (output / "source-output-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(json.dumps({"training": result["training"], "native_capture_validation": result["native_capture_validation"]}, indent=2))
