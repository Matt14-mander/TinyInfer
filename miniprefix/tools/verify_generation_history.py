"""Independent FP32 multi-turn full-page history reuse checks on real CPU/CUDA workers."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from miniprefix.bindings.python.client import NativeEngine


def verify(root,workers):
    import numpy as np
    import torch
    import transformers
    torch.set_num_threads(4)
    model=transformers.AutoModelForCausalLM.from_pretrained(root,local_files_only=True,
        torch_dtype=torch.float32,attn_implementation="eager").eval()
    def logits(prompt):
        with torch.inference_mode():
            ids=torch.tensor([prompt],dtype=torch.long)
            return model(ids,attention_mask=torch.ones_like(ids),use_cache=False).logits[0,-1].numpy()
    references={}
    def greedy(prompt,count,eos):
        key=(tuple(prompt),count,eos)
        if key not in references:
            growing=list(prompt);output=[]
            for _ in range(count):
                token=int(logits(growing).argmax());growing.append(token);output.append(token)
                if token==eos:break
            references[key]=output
        return references[key]
    records=[];numeric=[]
    for backend,worker in workers.items():
        for capacity in (0,32):
            with NativeEngine(worker,root/'model.mpf',backend,128,2,8,2,4,64,capacity) as e:
                def generate(name,prompt,count,reuse,eos=None):
                    before=e.stats();e.submit(name,prompt,count,eos);events=[];steps=[]
                    for _ in range(256):
                        if not e.pending:break
                        events+=e.step();steps.append(e.last_step_stats())
                    assert not e.pending,'no progress'
                    output=[v['token_id'] for v in events if v['token_id'] is not None]
                    assert output==greedy(prompt,count,eos),(backend,name,'output mismatch')
                    reasons=[v['finish_reason'] for v in events if v['finish_reason'] is not None]
                    assert reasons==['eos' if eos is not None and output[-1]==eos else 'length']
                    after=e.stats();actual=after['prefix_reused_tokens']-before['prefix_reused_tokens']
                    assert actual==(reuse if capacity else 0),(name,actual,reuse,capacity)
                    assert after['forward_tokens']-before['forward_tokens']==len(prompt)+len(output)-1-actual
                    assert sum(s['prefix_reused_tokens'] for s in steps)==actual
                    assert sum(s['prefill_tokens']+s['decode_tokens'] for s in steps)==len(prompt)+len(output)-1-actual
                    assert after['live_sequences']==0 and after['block_table_device_bytes']==0
                    if backend=='cuda':
                        assert after['kv_uploaded_bytes']==before['kv_uploaded_bytes']
                        assert after['weight_uploaded_bytes']==before['weight_uploaded_bytes']
                    records.append(dict(backend=backend,cache_pages=capacity,case=name,prompt=prompt,output=output,
                                        reused_tokens=actual,forward_tokens=after['forward_tokens']-before['forward_tokens']))
                    return output
                prompt=[1,504,3575]
                out=generate('turn1',prompt,7,0)
                prompt=prompt+out+[30,504,198]
                out=generate('turn2',prompt,5,8)
                prompt=prompt+out+[30,198]
                # Inspect restored page contents against independent full-prefix logits.
                restored=e.reuse_prefix('probe',prompt)
                actual=np.asarray(e.forward('probe',prompt[restored:]),dtype=np.float32)
                reference=logits(prompt);difference=np.abs(actual-reference)
                assert np.isfinite(actual).all() and np.all(difference<=2e-3+2e-4*np.abs(reference))
                assert int(actual.argmax())==int(reference.argmax())
                numeric.append(dict(backend=backend,cache_pages=capacity,reused_tokens=restored,
                                    checked_logits=len(actual),max_abs_error=float(difference.max())))
                e.release('probe')
                generate('turn3',prompt,3,16)
                e.clear_prefix_cache()
                base=[1,504,3575]
                final=generate('one-token',base,1,0)
                generate('uncomputed-final',base+final+[30],2,0)
                e.clear_prefix_cache()
                out=generate('aligned-seed',base,6,0)
                generate('aligned-hit',(base+out)[:8],2,4)
                e.clear_prefix_cache()
                eos=greedy(base,2,None)[-1]
                out=generate('eos-seed',base,8,0,eos)
                expected=(len(base)+len(out)-1)//4*4
                generate('after-eos',base+out+[30],2,expected)
                e.clear_prefix_cache()
                # Cancel after three sampled tokens: committed KV is prompt + first two outputs.
                e.submit('cancel-seed',base,8,None);cancelled=[]
                for _ in range(128):
                    cancelled += [v['token_id'] for v in e.step() if v['token_id'] is not None]
                    if len(cancelled)==3:break
                assert cancelled==greedy(base,3,None)
                assert e.cancel('cancel-seed')[0]['finish_reason']=='cancelled'
                generate('after-cancel',base+cancelled+[30],2,4)
                e.clear_prefix_cache()
                e.submit('live',base,6,None);live=[]
                for _ in range(128):
                    batch=e.step();live += [v['token_id'] for v in batch if v['token_id'] is not None]
                    if live:break
                assert len(live)==1
                fork=base+live+[30];before=e.stats();e.submit('fork',fork,3,None)
                outputs={'live':live,'fork':[]};finished={'live':0,'fork':0}
                for _ in range(256):
                    if not e.pending:break
                    for event in e.step():
                        name=event['request_id']
                        assert finished[name]==0
                        if event['token_id'] is not None:outputs[name].append(event['token_id'])
                        if event['finish_reason'] is not None:
                            assert event['finish_reason']=='length';finished[name]+=1
                assert not e.pending and finished=={'live':1,'fork':1}
                assert outputs['live']==greedy(base,6,None) and outputs['fork']==greedy(fork,3,None)
                shared=e.stats()['prefix_reused_tokens']-before['prefix_reused_tokens']
                assert shared==(4 if capacity else 0)
                records.append(dict(backend=backend,cache_pages=capacity,case='concurrent-fork',
                                    prompt=fork,output=outputs,reused_tokens=shared))
                e.clear_prefix_cache();s=e.stats()
                assert s['kv_live_blocks']==s['live_sequences']==s['block_table_device_bytes']==0
    return dict(passed=True,reference='Transformers CPU FP32 eager full-prefix; use_cache=False',
                torch=torch.__version__,transformers=transformers.__version__,records=records,logits=numeric,
                model_sha256=hashlib.sha256((root/'model.mpf').read_bytes()).hexdigest(),
                workers={k:hashlib.sha256(Path(v).read_bytes()).hexdigest() for k,v in workers.items()})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--cpu-worker',required=True);p.add_argument('--cuda-worker');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();workers={'cpu':a.cpu_worker}
    if a.cuda_worker:workers['cuda']=a.cuda_worker
    result=verify(a.model_dir,workers);a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,cases=len(result['records']),logits=sum(v['checked_logits'] for v in result['logits']))))
if __name__=='__main__':main()
