import inspect, sys, time, json
from pathlib import Path
import adk_submission
from adk_submission import VllmConfig, VllmServer
print("VllmConfig:", inspect.signature(VllmConfig), flush=True); print("VllmServer:", inspect.signature(VllmServer), flush=True)
model = sys.argv[1]
from adk_submission import discover_adapters
manifest = discover_adapters(Path(sys.argv[3])) if len(sys.argv) > 3 else None
ranks = [a.rank for a in (manifest.adapters.values() if manifest else []) if getattr(a, "rank", None)]
print("manifest:", manifest, flush=True)
cfg = VllmConfig(model=model, port=8000, host="127.0.0.1", tool_call_parser="gemma4", reasoning_parser="gemma4", max_model_len=32768, dtype="bfloat16", gpu_memory_utilization=0.85, enable_auto_tool_choice=True, enable_lora=bool(manifest and manifest.adapters), max_loras=max(1, len(manifest.adapters) if manifest else 1), max_lora_rank=(max(ranks) if ranks else 128), scheduler_reserve_full_isl=False, tensor_parallel_size=1, startup_timeout=60*20, reasoning_config={"reasoning_start_str": "<|channel>", "reasoning_end_str": "<channel|>"}, extra_args=["--served-model-name", sys.argv[2], "--chat-template", "PROJECT_ROOT/gemma4/analysis/matched-context-boss-v049/chat_template.matched.jinja"])
print("cfg:", cfg, flush=True)
srv = VllmServer(cfg, adapter_manifest=manifest); srv.start(); print("SERVER STARTED", flush=True)
while True: time.sleep(60)
