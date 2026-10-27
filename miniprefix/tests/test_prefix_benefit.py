import unittest
from miniprefix.benchmarks.prefix_benefit import validate_counters, paired_comparisons, sampling_order, SCENARIOS


class PrefixBenefitTests(unittest.TestCase):
    def test_each_workload_rotates_all_modes_independently_of_position(self):
        orders=[sampling_order(i) for i in range(3)]
        for order in orders:
            self.assertEqual(len(order),9)
            self.assertEqual(len(set(order)),9)
        for scenario in SCENARIOS:
            first_modes=[next(mode for name,mode in order if name==scenario) for order in orders]
            self.assertEqual(set(first_modes),{"off","cold","warm"})

    def test_hits_must_reduce_forward_and_gpu_input(self):
        before=dict(live_sequences=0,block_table_device_bytes=0,kv_live_blocks=0,kv_cache_only_blocks=0,
            kv_cached_blocks=0,prefix_reused_tokens=0,forward_tokens=0,prefix_cache_capacity=48,
            prefix_hits=0,prefix_misses=0,kv_reserved_bytes=160,kv_total_blocks=10,device_workspace_bytes=32,
            weight_uploaded_bytes=100,kv_uploaded_bytes=0,token_uploaded_bytes=0,device_allocations=3,
            prefix_evictions=0,peak_kv_bytes=0)
        after=dict(before,kv_live_blocks=2,kv_cache_only_blocks=2,kv_cached_blocks=2,
            prefix_reused_tokens=4,forward_tokens=6,prefix_hits=1,token_uploaded_bytes=24,
            device_allocations=4,peak_kv_bytes=48)
        workload=[dict(prompt_tokens=list(range(8)),max_new_tokens=3)]
        result=validate_counters(before,after,workload,"cuda","warm")
        self.assertEqual(result["uncached_forward_tokens"],10)
        self.assertEqual(result["actual_forward_tokens"],6)
        self.assertEqual(result["retained_cache_payload_bytes"],32)
        with self.assertRaisesRegex(RuntimeError,"skip actual"):
            validate_counters(before,dict(after,forward_tokens=10),workload,"cuda","warm")
        with self.assertRaisesRegex(RuntimeError,"residency"):
            validate_counters(before,dict(after,token_uploaded_bytes=40),workload,"cuda","warm")

    def test_pair_ratios_are_per_round_and_charge_prime(self):
        rows=[]
        for scenario in SCENARIOS:
            for run,off_time,on_time in ((1,100,50),(2,1000,250),(3,100,50)):
                for mode in ("off","cold","warm"):
                    time=off_time if mode=="off" else on_time
                    rows.append(dict(scenario=scenario,run=run,mode=mode,ttft_p50_ms=time,
                        output_tokens_per_second=1000/time,prime_inclusive_tokens_per_second=1000/(time+50) if mode=="warm" else 1000/time,
                        tpot_mean_ms=10))
        r=paired_comparisons(rows)[SCENARIOS[0]]["warm"]
        self.assertEqual([p["ttft_p50_speedup"] for p in r["pairs"]],[2,4,2])
        self.assertEqual(r["summary"]["ttft_p50_speedup"]["median"],2)
        self.assertLess(r["summary"]["prime_inclusive_throughput_speedup"]["median"],
                        r["summary"]["throughput_speedup"]["median"])


if __name__=="__main__":unittest.main()
