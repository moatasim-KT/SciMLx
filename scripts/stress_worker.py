"""Single stress-test worker — spawned as a subprocess by the concurrency test."""
import sys, time, uuid
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

worker_id = int(sys.argv[1])
from core.results_store import store

for i in range(10):
    store.append({
        "id": f"stress_{worker_id}_{i}_{uuid.uuid4().hex[:6]}",
        "parent_id": None,
        "timestamp": int(time.time()),
        "benchmark": "stress_test",
        "model": "FNO",
        "val_l2_rel": 0.1 + worker_id * 0.01 + i * 0.001,
        "memory_gb": 0.5,
        "status": "keep",
        "description": f"stress w={worker_id} i={i}",
        "commit": "test",
        "config": {"name": f"stress_{worker_id}_{i}"},
        "rationale": "", "conclusion": "", "diag": {},
    })
print(f"worker {worker_id} done")
