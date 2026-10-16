"""Bounded GPU controls and complete-model execution with new request payloads."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
import torch,triton
from baseline_persistent import recurrence as baseline
from flashrnn.flashrnn2.triton_persistent import recurrence
from flashrnn.flashrnn2.state_contract import begin_sequence
from flashrnn.flashrnn2.prepared import PreparedSequence
from flashrnn.flashrnn2.execution_policy import ExecutionPolicy,KernelConfig,shape_key
from fixture_math import compare,CELL_BUDGET,MODEL_BUDGET
from plan import CONFIGS,configs_for,chunk_lengths

BASE=Path(os.environ['FRNN_OPT_BASE']);OUT=BASE/'science'
DT={'float16':torch.float16,'bfloat16':torch.bfloat16}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.partial');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)
def array(x):return x.detach().float().cpu().numpy()
def same_bits(a,b):
    return a.shape==b.shape and a.dtype==b.dtype and np.array_equal(np.ascontiguousarray(a).view(np.uint8),np.ascontiguousarray(b).view(np.uint8))
def raw(p,values):
    np.savez_compressed(p,**{k:array(v) if isinstance(v,torch.Tensor) else v for k,v in values.items()})
    return dict(file=str(p),bytes=p.stat().st_size,sha256=sha(p))
def identity():
    return dict(gpu=torch.cuda.get_device_name(),capability=list(torch.cuda.get_device_capability()),
                uuid=os.environ['FRNN_OPT_GPU_UUID'],triton=triton.__version__,source_manifest_sha256=sha(BASE/'source.manifest.json'))
def inputs(item,split):
    r=item['splits'][str(split)];assert sha(r['file'])==r['sha256'];z=dict(np.load(r['file']))
    dt=DT[item['case']['dtype']]
    return z,tuple(torch.tensor(z[k],device='cuda',dtype=dt) for k in ('wx','r','bias','initial'))
def public_call(values,cell,*,mode='full',config=None,snapshot=torch.float32):
    wx,r,b,initial=values;cfg=config or KernelConfig()
    context=begin_sequence(initial,cell,wx.dtype)
    if cfg.family=='simt':
        p=PreparedSequence(*values,cell,config=cfg,output_mode=mode,snapshot_dtype=snapshot)
        return p.run()
    return recurrence(*values,cell,return_compute_carry=True,sequence_context=context,
        snapshot_dtype=snapshot,output_mode=mode,num_warps=cfg.warps,weight_operand=cfg.weight_operand)
def legacy_call(values,cell,snapshot=torch.float32):
    return baseline(*values,cell,return_compute_carry=True,sequence_context=begin_sequence(values[3],cell,values[0].dtype),snapshot_dtype=snapshot)
def legacy_chunks(values,cell):
    wx,r,b,initial=values;ctx=begin_sequence(initial,cell,wx.dtype);y=[];start=0;carry=initial
    for n in chunk_lengths(wx.shape[1]):
        h,carry=baseline(wx[:,start:start+n],r,b,carry,cell,snapshot_dtype=torch.float32,
                         return_compute_carry=True,sequence_context=ctx)
        y.append(h);ctx=ctx.advance(n);start+=n
    return torch.cat(y,dim=2),carry
def cell_functions(item,split,selected=None,graphs=False):
    z,values=inputs(item,split);c=item['case'];cell=c['cell'];plans={}
    for name in configs_for(c):plans[name]=PreparedSequence(*values,cell,config=CONFIGS[name],snapshot_dtype=torch.float32)
    for name in ('hidden','last'):
        plans[name]=PreparedSequence(*values,cell,output_mode=name,snapshot_dtype=torch.float32)
    if selected is not None:plans['selected']=PreparedSequence(*values,cell,config=selected,snapshot_dtype=torch.float32)
    if (c['batch'],c['steps'],c['width'],c['heads']) in ((3,17,48,2),(16,128,64,2),(2,64,128,4)):
        plans['prepared_chunks']=PreparedSequence(*values,cell,chunks=chunk_lengths(c['steps']),snapshot_dtype=torch.float32)
    fns={name:p.run for name,p in plans.items()};fns['legacy']=lambda:legacy_call(values,cell)
    if 'prepared_chunks' in plans:
        fns['legacy_chunks']=lambda:legacy_chunks(values,cell)
        if graphs:
            p=PreparedSequence(*tuple(x.clone() for x in values),cell,chunks=chunk_lengths(c['steps']),snapshot_dtype=torch.float32)
            p.capture();plans['graph_chunks']=p;fns['graph_chunks']=p.run
    return z,values,plans,fns
def expected_cell(z,arm):
    h=z['reference']
    if arm=='hidden':h=h[0]
    if arm=='last':h=h[0][:,-1:]
    return dict(history=h,carry=z['carry'])
def check_cell(values,z,arm):
    return {k:compare(array(v),expected_cell(z,arm)[k],CELL_BUDGET) for k,v in zip(('history','carry'),values)}
def passed(checks):return all(r['pass_'] for r in checks.values())
def timing(fn,n):
    torch.cuda.synchronize();start=time.perf_counter_ns()
    for _ in range(n):fn()
    torch.cuda.synchronize();return (time.perf_counter_ns()-start)/1e6/n
def pairs(a,b,blocks=20,prepare=None):
    for _ in range(3):a();b()
    trial=timing(a,1);n=max(1,min(10,int(5/max(trial,.01))))
    result=[]
    for i in range(blocks):
        order='ABBA' if i%2==0 else 'BAAB';vals={'A':[],'B':[]}
        for arm in order:
            if prepare is not None:prepare(i)
            vals[arm].append(timing(a if arm=='A' else b,n))
        result.append(dict(block=i,iterations=n,order=order,
            payload_start=i%3 if prepare is not None else None,
            a_ms=float(np.mean(vals['A'])),b_ms=float(np.mean(vals['B']))))
    return result


class Model:
    def __init__(self,item,policy):
        assert sha(item['file'])==item['sha256'];self.item=item;self.z=dict(np.load(item['file']))
        self.policy=policy;self.identity=identity();self.payloads=[torch.from_numpy(self.z[f'ids{i}']) for i in range(3)]
        self.weights={k:torch.tensor(v,device='cuda',dtype=torch.float16) for k,v in self.z.items()
                      if k in ('embedding','readout') or k[0] in ('w','r','b') and k[-1].isdigit()}
        c=item;self.b,self.t,self.h,self.d=(c[k] for k in ('batch','steps','heads','width'));e=self.h*self.d
        self.cfg=policy.select(c['cell'],'float16',self.b,self.t,self.h,self.d,identity=self.identity)
        # Dispatch-only keeps the original MMA arithmetic; SIMT is its own arm.
        self.mma_cfg=self.cfg if self.cfg.family=='mma' else KernelConfig()
        self.ids=torch.empty((self.b,self.t),device='cuda',dtype=torch.long)
        self.activation=torch.empty((self.b,self.t,e),device='cuda',dtype=torch.float16)
        self.projection=[];self.sequence=[]
        for l in range(4):
            wx=torch.empty((self.b,self.t,4,self.h,self.d),device='cuda',dtype=torch.float16)
            initial=torch.zeros((2 if c['cell']=='lstm' else 4,self.b,1,self.h,self.d),device='cuda',dtype=torch.float16)
            p=PreparedSequence(wx,self.weights[f'r{l}'],self.weights[f'b{l}'],initial,c['cell'],
                               config=self.cfg,output_mode='last' if l==3 else 'hidden')
            self.projection.append(wx);self.sequence.append(p)
        self.logits=torch.empty((self.b,512),device='cuda',dtype=torch.float16)
        self.graphs={};self.graph_results={};self.counters={a:0 for a in ('legacy','dispatch','hidden','small','prepared','legacy_graph','optimized_graph')}

    def eager_gpu(self,ids,arm,trace=False):
        c=self.item;e=self.h*self.d;s=2 if c['cell']=='lstm' else 4
        x=self.weights['embedding'][ids];traces={}
        for l in range(4):
            wx=torch.nn.functional.linear(x,self.weights[f'w{l}'].reshape(4*e,e)).reshape(self.b,self.t,4,self.h,self.d)
            initial=torch.zeros((s,self.b,1,self.h,self.d),device='cuda',dtype=torch.float16)
            vals=(wx,self.weights[f'r{l}'],self.weights[f'b{l}'],initial)
            mode='full' if arm in ('legacy','dispatch') else ('last' if l==3 else 'hidden')
            if arm=='legacy':y,carry=legacy_call(vals,c['cell'],snapshot=None)
            else:y,carry=public_call(vals,c['cell'],mode=mode,config=self.cfg if arm=='small' else self.mma_cfg,snapshot=None)
            h=y[0] if mode=='full' else y
            if trace:
                traces[f'hidden{l}']=h
                traces[f'carry{l}']=carry
            x=h.reshape(self.b,1 if mode=='last' else self.t,e)
        return x[:,-1]@self.weights['readout'],traces

    def prepared_gpu(self,trace=False):
        e=self.h*self.d
        torch.index_select(self.weights['embedding'],0,self.ids.reshape(-1),out=self.activation.view(self.b*self.t,e))
        x=self.activation;traces={}
        for l,p in enumerate(self.sequence):
            torch.mm(x.reshape(self.b*self.t,e),self.weights[f'w{l}'].reshape(4*e,e).T,
                     out=self.projection[l].view(self.b*self.t,4*e))
            h,carry=p.run()
            if trace:traces[f'hidden{l}']=h;traces[f'carry{l}']=carry
            x=h.reshape(self.b,1 if l==3 else self.t,e)
        torch.mm(x[:,-1],self.weights['readout'],out=self.logits)
        return self.logits,traces

    def capture(self,arm):
        if arm in self.graphs:return
        self.ids.copy_(self.payloads[0][:self.b]);stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        fn=(lambda:self.eager_gpu(self.ids,'legacy',True)) if arm=='legacy_graph' else (lambda:self.prepared_gpu(True))
        with torch.cuda.stream(stream):
            for _ in range(3):fn()
        torch.cuda.current_stream().wait_stream(stream)
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):result=fn()
        self.graphs[arm]=g;self.graph_results[arm]=result

    def run(self,arm,payload=None,save=False):
        if payload is None:payload=self.counters[arm]%3;self.counters[arm]+=1
        if arm.endswith('graph'):self.capture(arm)
        answers=[];traces={}
        for start in range(0,32,self.b):
            host=self.payloads[payload][start:start+self.b]
            if arm in ('prepared','legacy_graph','optimized_graph'):
                self.ids.copy_(host)
                if arm.endswith('graph'):self.graphs[arm].replay();logits,layers=self.graph_results[arm]
                else:logits,layers=self.prepared_gpu(save)
            else:logits,layers=self.eager_gpu(host.to('cuda'),arm,save)
            # CPU delivery is inside every measured complete-call arm.
            answers.append(logits.cpu())
            if save:
                for k,v in layers.items():traces.setdefault(k,[]).append(v.clone())
        if not save:return answers
        return dict(logits=torch.cat(answers),**{k:torch.cat(v,dim=1 if k.startswith('carry') else 0) for k,v in traces.items()})

    def expected(self,payload,values):
        z=self.z;expected={}
        for k,v in values.items():
            key=f'logits{payload}' if k=='logits' else f'{k[:-1]}{payload}_{k[-1]}'
            x=z[key]
            if k.startswith('hidden') and v.shape[1]==1:x=x[:,-1:]
            expected[k]=x
        return expected

    def close(self):
        torch.cuda.synchronize()
        for g in self.graphs.values():g.reset()
        for p in self.sequence:p.close()
        self.graphs.clear();self.graph_results.clear()
