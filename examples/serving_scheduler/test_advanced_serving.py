"""Preemption/replay, immutable prefix sharing/COW, and optional real CUDA."""
import os
import random
import unittest
try:
    from .backend import Request, WorkItem
    from .tinyinfer_backend import TinyInferBackend
    from .realtime import serve
    from .metrics import summarize
except ImportError:
    from backend import Request, WorkItem
    from tinyinfer_backend import TinyInferBackend
    from realtime import serve
    from metrics import summarize


class AdvancedServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker = os.environ.get("TINYINFER_SERVING_WORKER")
        if not cls.worker:
            raise unittest.SkipTest("actual CPU worker required")

    def backend(self, capacity=8, sharing=False, size=2):
        return TinyInferBackend(self.worker, size, "paged", capacity, "direct", sharing)

    def test_prefix_hit_skips_compute_and_survives_source_release(self):
        # The deterministic prompt seeds for A/a differ by 32, hence equal tokens.
        a, b = Request("A",0,9,2), Request("a",0,9,2)
        with self.backend(16,True,4) as backend:
            self.assertEqual(backend.prompt(a),backend.prompt(b))
            expected = backend.reference(b)
            backend.step([WorkItem(a,0)])
            source = backend.block_table("A")
            backend.release("A")
            self.assertGreater(backend.stats()["prefix_cache_entries"],0)
            before = backend.stats()["forward_tokens"]
            backend.step([WorkItem(b,0)])
            stats = backend.stats()
            self.assertEqual(stats["forward_tokens"]-before,1)
            self.assertEqual(stats["prefix_reused_tokens"],8)
            self.assertEqual(backend.block_table("a")[:2],source[:2])
            self.assertEqual(backend.outputs["a"],[expected[0]])
            backend.step([WorkItem(b,1,9)])
            self.assertEqual(backend.outputs["a"],expected)
            backend.clear()
            self.assertEqual(backend.stats()["live_physical_blocks"],0)

    def test_shared_partial_block_cow_different_suffixes(self):
        class Prompts(TinyInferBackend):
            @staticmethod
            def prompt(request):
                return [1,2,3] if request.request_id=="source" else [1,2,3,9,10]
        a,b=Request("source",0,3,3),Request("peer",0,5,2)
        with Prompts(self.worker,4,"paged",8,"direct",True) as backend:
            expected={r.request_id:backend.reference(r) for r in (a,b)}
            backend.step([WorkItem(a,0)])
            source=backend.block_table("source")
            backend.step([WorkItem(b,0)])
            self.assertNotEqual(backend.block_table("peer")[0],source[0])
            self.assertEqual(backend.stats()["prefix_reused_tokens"],3)
            self.assertGreater(backend.stats()["cow_block_copies"],0)
            backend.step([WorkItem(a,1,3),WorkItem(b,1,5)])
            backend.release("peer")
            backend.step([WorkItem(a,2,3)])
            for r in (a,b):self.assertEqual(backend.outputs[r.request_id],expected[r.request_id])
            backend.clear()
            self.assertEqual(backend.stats()["free_physical_blocks"],8)

    def test_growth_preempts_and_replays_without_duplicate_delivery(self):
        requests=[Request("a",0,1,5),Request("b",0,1,5)]
        with self.backend(3) as backend:
            expected={r.request_id:backend.reference(r) for r in requests}
            run=serve(requests,backend,max_batch_size=2,kv_capacity_blocks=3,kv_block_size=2,
                      token_budget=3,prefill_chunk_size=2,kv_allocation="dynamic")
            report=summarize(run)
            self.assertEqual(len(run.results),2)
            for result in run.results:
                self.assertEqual(result.tokens,expected[result.request.request_id])
                self.assertEqual(len(result.token_times_ms),5)
                self.assertEqual(sorted(result.token_times_ms),result.token_times_ms)
            self.assertTrue(any(e["event"]=="preempt" and e["delivered_tokens"]>0 for e in run.admission_events))
            self.assertGreater(run.backend_stats["recompute_tokens"],0)
            self.assertEqual(run.backend_stats["recompute_tokens"],report["recompute_input_tokens"])
            self.assertEqual(run.backend_stats["forward_tokens"],10+report["recompute_input_tokens"])
            self.assertLessEqual(report["peak_step_input_tokens"],3)
            self.assertEqual(run.backend_stats["live_physical_blocks"],0)

    def test_dynamic_random_budgets_with_and_without_sharing(self):
        rng=random.Random(20261022)
        requests=[Request("r%d"%i,(i%3)*.1,rng.randint(1,4),rng.randint(2,6)) for i in range(9)]
        requests.append(Request("too-big",0,20,1))
        for sharing in (False,True):
            with self.backend(6,sharing) as backend:
                expected={r.request_id:backend.reference(r) for r in requests}
                for budget in (1,3,7,None):
                    run=serve(requests,backend,max_batch_size=3,kv_capacity_blocks=6,kv_block_size=2,
                              token_budget=budget,prefill_chunk_size=2 if budget else None,kv_allocation="dynamic")
                    self.assertEqual(len(run.results),9)
                    self.assertEqual(len(run.rejections),1)
                    for result in run.results:self.assertEqual(result.tokens,expected[result.request.request_id])
                    stats=run.backend_stats
                    scheduled=sum(s["input_tokens"] for s in run.trace)
                    self.assertEqual(stats["forward_tokens"]+stats["prefix_reused_tokens"],scheduled)
                    self.assertEqual(stats["linear_matmul_calls"],7*len(run.trace))
                    self.assertEqual(stats["live_physical_blocks"],0)
                    self.assertEqual(stats["prefix_cache_entries"],0)
                    if budget:self.assertTrue(all(s["input_tokens"]<=budget for s in run.trace))

    def test_dynamic_failure_cleanup_and_invalid_modes(self):
        with self.backend(4,True) as backend:
            for mode,policy in (("static","wait"),("continuous","reject")):
                with self.assertRaises(ValueError):
                    serve([],backend,mode=mode,kv_policy=policy,kv_capacity_blocks=4,kv_block_size=2,kv_allocation="dynamic")
            original=backend.step
            calls=[0]
            def fail(work):
                result=original(work);calls[0]+=1
                if calls[0]==3:raise RuntimeError("failure after shared/replay step")
                return result
            backend.step=fail
            with self.assertRaisesRegex(RuntimeError,"failure after"):
                serve([Request("a",0,2,5),Request("b",0,2,5)],backend,kv_capacity_blocks=4,kv_block_size=2,
                      token_budget=3,kv_allocation="dynamic")
            self.assertEqual(backend.stats()["live_physical_blocks"],0)
            self.assertEqual(backend.stats()["prefix_cache_entries"],0)
            backend.step=original
            run=serve([Request("a",0,2,2)],backend,kv_capacity_blocks=4,kv_block_size=2,kv_allocation="dynamic")
            self.assertEqual(len(run.results),1)


class CudaServingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.worker=os.environ.get("TINYINFER_SERVING_CUDA_WORKER")
        if not cls.worker:raise unittest.SkipTest("optional real CUDA serving worker required")

    def test_cuda_logits_mixed_batches_prefix_and_replay(self):
        a,b=Request("A",0,9,3),Request("a",0,9,3)
        with TinyInferBackend(self.worker,4,"paged",8,"cuda",True) as backend:
            expected={r.request_id:backend.reference(r) for r in (a,b)}
            for work in ([WorkItem(a,0)],[WorkItem(a,1,9),WorkItem(b,0)],
                         [WorkItem(b,1,9),WorkItem(a,2,9)],[WorkItem(b,2,9)]):
                backend.step(work)
                actual=backend.last_step_logits()
                for i,item in enumerate(work):
                    prefix=backend.prompt(item.request)+backend.outputs[item.request.request_id][:-1]
                    if item.is_prefill:prefix=backend.prompt(item.request)
                    oracle=backend.reference_logits(prefix)
                    for x,y in zip(actual[i],oracle):self.assertLessEqual(abs(x-y),2e-5+2e-5*abs(y))
            for r in (a,b):self.assertEqual(backend.outputs[r.request_id],expected[r.request_id])
            stats=backend.stats()
            self.assertEqual(stats["cuda_kernel_launches"],4)
            self.assertEqual(stats["cuda_attention_queries"],stats["forward_tokens"])
            self.assertGreater(stats["prefix_reused_tokens"],0)
            backend.clear()
            self.assertEqual(backend.stats()["live_physical_blocks"],0)
        with TinyInferBackend(self.worker,2,"paged",3,"cuda",True) as backend:
            requests=[Request("a",0,1,5),Request("b",0,1,5)]
            expected={r.request_id:backend.reference(r) for r in requests}
            run=serve(requests,backend,kv_capacity_blocks=3,kv_block_size=2,token_budget=3,kv_allocation="dynamic")
            self.assertGreater(run.backend_stats["recompute_tokens"],0)
            self.assertEqual(run.backend_stats["cuda_kernel_launches"],len(run.trace))
            for result in run.results:self.assertEqual(result.tokens,expected[result.request.request_id])

    def test_cuda_long_ragged_contexts_and_reset(self):
        for size in (1,3,7):
            requests=[Request("short",0,1,2),Request("long",0,129,2)]
            with TinyInferBackend(self.worker,size,"paged",140,"cuda") as backend:
                expected={r.request_id:backend.reference(r) for r in requests}
                backend.step([WorkItem(r,0) for r in requests])
                backend.step([WorkItem(r,1,r.prompt_tokens) for r in reversed(requests)])
                for i,r in enumerate(reversed(requests)):
                    prefix=backend.prompt(r)+backend.outputs[r.request_id][:-1]
                    oracle=backend.reference_logits(prefix)
                    for x,y in zip(backend.last_step_logits()[i],oracle):
                        self.assertLessEqual(abs(x-y),2e-5+2e-5*abs(y))
                    self.assertEqual(backend.outputs[r.request_id],expected[r.request_id])
                self.assertEqual(backend.stats()["cuda_kernel_launches"],2)
                backend.reset()
                self.assertEqual(backend.stats()["cuda_kernel_launches"],0)
                self.assertEqual(backend.stats()["cuda_uploaded_bytes"],0)
                self.assertEqual(backend.stats()["live_physical_blocks"],0)


if __name__=="__main__":unittest.main()
