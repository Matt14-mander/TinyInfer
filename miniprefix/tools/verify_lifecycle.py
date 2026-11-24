"""Real-model lifecycle checks through the CPU/CUDA worker and Python client."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine


def rejected(operation):
    try:
        operation()
    except (ValueError, RuntimeError):
        return
    raise AssertionError('operation unexpectedly accepted')


def verify(worker, model, backend):
    with NativeEngine(worker, model, backend, context=32, batch_size=2,
                      token_budget=8, chunk_size=4, block_tokens=2, kv_blocks=16,
                      prefix_cache_blocks=8, max_waiting_requests=2, max_event_slots=16,
                      history_capacity=1, history_ttl_ms=20) as engine:
        engine.submit('expired', [1, 2, 3], 4, None, timeout_ms=1)
        time.sleep(.02)  # Expires before any native forward; no timing upper-bound assertion.
        events=engine.step()
        assert len(events)==1 and events[0]['finish_reason']=='timeout' and events[0]['token_id'] is None
        assert engine.stats()['forward_tokens']==0
        time.sleep(.03)
        engine.submit('expired', [1, 2, 3], 1, None)
        engine.submit('cancel', [1, 2, 3], 1, None)
        rejected(lambda: engine.submit('retry', [1], 1, None))
        assert 'retry' not in engine.ids
        events=engine.cancel('cancel')
        assert len(events)==1 and events[0]['finish_reason']=='cancelled'
        engine.submit('retry', [1], 1, None)
        while engine.pending:
            assert all(e['finish_reason'] in (None,'length') for e in engine.step())
        for i in range(24):
            engine.submit('cycle'+str(i), [1, 2, 3], 1, None)
            events=engine.step()
            assert len(events)==1 and events[0]['finish_reason']=='length'
            assert len(engine.ids)<=1
            stats=engine.lifecycle_stats()
            assert stats['event_slots']==stats['waiting']==stats['active']==0
            assert stats['tracked_ids']<=1 and stats['history_ids']<=1
        engine.clear_prefix_cache()
        assert engine.stats()['kv_live_blocks']==0
        rejected(lambda: engine.submit('too-large-event-budget', [1], 16, None))
        assert not engine.pending
        return dict(backend=backend, passed=True, lifecycle=engine.lifecycle_stats(),
                    worker_sha256=hashlib.sha256(Path(worker).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--cpu-worker', required=True)
    parser.add_argument('--cuda-worker')
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    results=[verify(args.cpu_worker,args.model,'cpu')]
    if args.cuda_worker: results.append(verify(args.cuda_worker,args.model,'cuda'))
    args.output.write_text(json.dumps(dict(passed=True,results=results,
        model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest()),indent=2)+'\n')
    print('Real-model lifecycle checks passed.')


if __name__=='__main__':
    main()
