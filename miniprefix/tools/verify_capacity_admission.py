"""Real CPU/CUDA bounded-capacity admission regression (explicit local model required)."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine


def drain(engine, limit=100):
    result=[]
    for _ in range(limit):
        if not engine.pending:
            return result
        result.extend(engine.step())
    raise AssertionError("capacity wait made no bounded progress")


def verify(worker, model, backend):
    prompt=[1,504,3575]
    def open_engine(pages):
        return NativeEngine(worker, model, backend, context=32, batch_size=2,
                            token_budget=2, chunk_size=1, block_tokens=2,
                            kv_blocks=pages, prefix_cache_blocks=3)
    with open_engine(32) as reference:
        reference.submit("reference",prompt,4,None)
        expected=[e["token_id"] for e in drain(reference) if e["token_id"] is not None]
    with open_engine(3) as engine:
        # Cache-only pages are reclaimable, not a reason to deny admission.
        engine.forward("seed",prompt+[4649]);engine.cache_prefix("seed");engine.release("seed")
        engine.submit("impossible",prompt,5,None) # needs 7 KV tokens / 4 pages
        engine.submit("a",prompt,4,None);engine.submit("b",prompt,4,None)
        events=drain(engine)
        for name in ("a","b"):
            actual=[e["token_id"] for e in events if e["request_id"]==name and e["token_id"] is not None]
            assert actual==expected,(backend,name,actual,expected)
            assert sum(e["request_id"]==name and e["finish_reason"]=="length" for e in events)==1
        failures=[e for e in events if e["finish_reason"]=="error"]
        assert len(failures)==1 and failures[0]["request_id"]=="impossible"
        assert engine.stats()["prefix_hits"]>0
        engine.submit("cancel-active",prompt,4,None);engine.submit("cancel-waiting",prompt,4,None)
        engine.step()
        assert engine.cancel("cancel-waiting")[0]["finish_reason"]=="cancelled"
        assert engine.cancel("cancel-active")[0]["finish_reason"]=="cancelled"
        # Early EOS releases the entire remaining logical reservation.
        engine.submit("eos",prompt,4,expected[0]);engine.submit("after-eos",prompt,4,None)
        eos_events=drain(engine)
        assert sum(e["request_id"]=="eos" for e in eos_events)==1
        assert next(e for e in eos_events if e["request_id"]=="eos")["finish_reason"]=="eos"
        assert [e["token_id"] for e in eos_events if e["request_id"]=="after-eos"]==expected
        engine.clear_prefix_cache()
        engine.forward("pinned",[1]*6)
        engine.submit("blocked",[1],1,None)
        blocked=drain(engine);assert len(blocked)==1 and blocked[0]["finish_reason"]=="error"
        engine.release("pinned")
        engine.submit("protected",prompt,4,None);engine.step()
        try:
            engine.forward("intruder",[1])
        except RuntimeError as error:
            assert "reserved" in str(error)
        else:
            raise AssertionError("diagnostics stole reserved capacity")
        engine.release("intruder")
        assert all(e["finish_reason"]!="error" for e in drain(engine))
        engine.clear_prefix_cache()
        stats=engine.stats();assert stats["live_sequences"]==stats["kv_live_blocks"]==0
    return dict(backend=backend,passed=True,expected_tokens=expected,stats=stats,
                worker_sha256=hashlib.sha256(Path(worker).read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",type=Path,required=True)
    parser.add_argument("--cpu-worker",required=True)
    parser.add_argument("--cuda-worker")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    records=[verify(args.cpu_worker,args.model,"cpu")]
    if args.cuda_worker:records.append(verify(args.cuda_worker,args.model,"cuda"))
    result=dict(passed=True,model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),records=records)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result))

if __name__=="__main__":main()
