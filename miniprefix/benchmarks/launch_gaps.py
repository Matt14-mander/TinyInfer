"""Summarize gaps inside synchronized, single-stream MiniPrefix forwards."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
from statistics import median


def family(name):
    for op in ('embedding', 'rms', 'rope', 'append', 'attention', 'add', 'swiglu', 'greedy'):
        if op + '_kernel' in name:
            return op
    if 'splitKreduce' in name:
        return 'splitK-reduce'
    if 'gemm' in name.lower():
        return 'GEMM'
    return name


def summarize(rows):
    calls = defaultdict(list)
    for row in rows:
        calls[row['mode'], row['phase'], int(row['call'])].append(row)
    groups = defaultdict(list)
    for (mode, phase, _), call in calls.items():
        if 'embedding_kernel' not in call[0]['name'] or len({k['stream'] for k in call}) != 1:
            raise ValueError('incomplete or multi-stream forward')
        previous_end = int(call[0]['start_ns'])
        span = int(call[-1]['end_ns']) - previous_end
        gaps, active, host_late, unresolved = [], 0, 0, 0
        transitions = Counter()
        for i, k in enumerate(call):
            start, end, api = (int(k[field]) for field in ('start_ns', 'end_ns', 'api_start_ns'))
            if start < previous_end or end <= start:
                raise ValueError('invalid kernel timestamps')
            active += end - start
            if i:
                gap = start - previous_end
                gaps.append(gap)
                transitions[family(call[i-1]['name']) + ' -> ' + family(k['name'])] += gap
                if api:
                    host_late += min(gap, max(0, api - previous_end))
                else:
                    unresolved += 1
            previous_end = end
        assert active + sum(gaps) == span
        groups[mode, phase].append(dict(span=span, active=active, gaps=gaps, host_late=host_late,
                                        unresolved=unresolved, transitions=transitions,
                                        sample_ns=sum(int(k['end_ns'])-int(k['start_ns']) for k in call
                                                      if family(k['name']) == 'greedy'),
                                        d2h_ns=int(call[0]['d2h_ns']), call_ns=int(call[0]['call_ns']),
                                        d2h_bytes=int(call[0]['d2h_bytes'])))
    result = []
    for (mode, phase), traces in groups.items():
        span = sum(t['span'] for t in traces)
        active = sum(t['active'] for t in traces)
        gaps = sorted(g for t in traces for g in t['gaps'])
        transitions = sum((t['transitions'] for t in traces), Counter())
        result.append(dict(mode=mode, phase=phase, calls=len(traces),
                           kernels_per_call=median(len(t['gaps'])+1 for t in traces),
                           span_ms_median=median(t['span'] for t in traces)/1e6,
                           append_ms_median=median(t['call_ns'] for t in traces)/1e6,
                           d2h_us_median=median(t['d2h_ns'] for t in traces)/1000,
                           d2h_bytes=median(t['d2h_bytes'] for t in traces),
                           greedy_us_median=median(t['sample_ns'] for t in traces)/1000,
                           active_pct=100*active/span, gap_pct=100*sum(gaps)/span,
                           gap_p50_us=median(gaps)/1000, gap_p95_us=gaps[math.ceil(len(gaps)*.95)-1]/1000,
                           api_late_pct_of_gaps=100*sum(t['host_late'] for t in traces)/sum(gaps) if sum(gaps) else 0,
                           unmatched_api=sum(t['unresolved'] for t in traces),
                           top_gap_transitions_ms=[(k,v/1e6) for k,v in transitions.most_common(3)]))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('traces', nargs='+', type=Path)
    results = []
    for path in parser.parse_args().traces:
        opener = gzip.open if path.suffix == '.gz' else open
        with opener(path, 'rt', encoding='utf-8-sig', newline='') as stream:
            results.extend(summarize(csv.DictReader(stream, delimiter='\t')))
    print(json.dumps(results, indent=2))
