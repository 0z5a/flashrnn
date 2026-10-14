"""Independent NumPy recurrence and predeclared generic numerical budgets."""
import numpy as np

CELL_BUDGET=dict(atol=.002,rtol=.02,max_abs=.003)
MODEL_BUDGET=dict(atol=.015,rtol=.03,max_abs=.03)
def round_input(x,dtype):
    x=np.asarray(x,dtype=np.float32)
    if dtype=='float16':return x.astype(np.float16).astype(np.float32)
    if dtype=='bfloat16':
        bits=x.view(np.uint32)
        return ((bits+np.uint32(0x7fff)+((bits>>16)&1))&np.uint32(0xffff0000)).view(np.float32)
    raise ValueError(dtype)

def reference(wx,r,b,initial,cell,dtype,round_state=False):
    """FP64 pointwise/accumulation; recurrent hidden rounds to input dtype."""
    state=initial[:,:,0].astype(np.float64).copy();history=[]
    global_zero=cell=='slstm' and bool(np.all(state[2]==0))
    for t in range(wx.shape[1]):
        hidden=round_input(state[0],dtype).astype(np.float64)
        ry=np.einsum('bhp,ghdp->bghd',hidden,r.astype(np.float64),optimize=False)
        i,f,z,o=np.moveaxis(wx[:,t].astype(np.float64)+ry+b.astype(np.float64),1,0)
        sig=lambda x:1/(1+np.exp(-x))
        if cell=='lstm':
            c=sig(f)*state[1]+sig(i)*np.tanh(z)
            state=np.stack((sig(o)*np.tanh(c),c))
        else:
            logf=np.minimum(f,0)-np.log1p(np.exp(-np.abs(f)))+state[3]
            m=i if t==0 and global_zero else np.maximum(i,logf)
            ig,fg=np.exp(i-m),np.exp(logf-m)
            c=fg*state[1]+ig*np.tanh(z)
            n=np.maximum(fg*state[2]+ig,1)
            state=np.stack((sig(o)*c/n,c,n,m))
        if round_state:state=round_input(state,dtype).astype(np.float64)
        history.append(state.copy())
    return np.stack(history,axis=2),state[:,:,None]

def compare(actual,expected,budget=CELL_BUDGET):
    assert actual.shape==expected.shape
    diff=np.abs(actual.astype(np.float64)-expected.astype(np.float64))
    limit=budget['atol']+budget['rtol']*np.abs(expected)
    return dict(pass_=bool(np.isfinite(actual).all() and np.all(diff<=limit) and diff.max()<=budget['max_abs']),
        max_abs=float(diff.max()),failed_elements=int(np.count_nonzero((diff>limit)|(diff>budget['max_abs']))),
        worst_normalized=float((diff/limit).max()),bitwise=bool(np.array_equal(actual,expected)))

def domain_cases():
    core=[(16,8),(17,2),(32,4),(48,2),(64,2),(65,2),(96,1),(128,1),(64,1),(64,8)]
    shapes=[(3,17,d,h) for d,h in core]+[(1,1,64,1),(1,1024,64,1),(32,128,64,2),(16,512,128,1)]
    return [dict(family='domain',cell=c,dtype=dt,batch=b,steps=t,width=d,heads=h,mode='warm')
        for c in ('lstm','slstm') for dt in ('float16','bfloat16') for b,t,d,h in shapes]

def boundary_cases():
    return [dict(family='boundary',cell=c,dtype=dt,batch=b,steps=t,width=d,heads=h,mode=mode)
        for c in ('lstm','slstm') for dt in ('float16','bfloat16')
        for b,t,d,h in ((3,17,48,2),(16,128,64,1),(2,64,128,4))
        for mode in (('warm',) if c=='lstm' else ('zero','mixed','warm'))]

def case_name(case):
    return '{family}-{cell}-{dtype}-b{batch}-t{steps}-d{width}-h{heads}-{mode}'.format(**case)

def make_fixture(case):
    seed=20261013+case['batch']*10000+case['steps']*100+case['width']+case['heads']*17
    rng=np.random.default_rng(seed);b,t,d,h=(case[k] for k in ('batch','steps','width','heads'))
    dt=case['dtype'];s=2 if case['cell']=='lstm' else 4
    wx=round_input(rng.normal(0,.1,(b,t,4,h,d)),dt)
    r=round_input(rng.normal(0,.1/d**.5,(4,h,d,d)),dt)
    bias=round_input(rng.normal(0,.03,(4,h,d)),dt)
    initial=round_input(rng.normal(0,.05,(s,b,1,h,d)),dt)
    if case['cell']=='slstm':
        initial[2]=round_input(np.abs(initial[2])+1,dt)
        if case['mode']=='zero':initial.fill(0)
        if case['mode']=='mixed':
            initial[:,0,:,0]=0;initial[3,0,:,0]=.75
    history,final=reference(wx,r,bias,initial,case['cell'],dt)
    return dict(wx=wx,r=r,bias=bias,initial=initial,reference_fp64=history,reference_final_fp64=final,
        reference_snapshot=round_input(history,dt))
