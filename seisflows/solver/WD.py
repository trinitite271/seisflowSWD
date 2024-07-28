import numpy as np


def RTp_cuda(seismo_v1,ii,df,dt,np,vmin,vmax,fmin,fmax,a,b,dg,offset,m,M):
    ii = ii+1
    nend=m+offset+M*ii-M
    if nend>b:
        nend=b
    uxt=seismo_v1[:,(M*ii-M)+m:nend].T
    offset=nend-(M*ii-M+1+m)+1
    # print(nend,offset)
    x=m+np.linspace(0,offset-1,offset)*dg
    x = x.view(1,-1).to('cuda')
    ccn=int(np.fix(np.tensor(1/df/dt)))
    d=np.fft.fft(uxt,n=ccn,dim=1)
    lf = int(np.round(np.tensor(fmin/df))+1)
    nf = int(np.round(np.tensor(fmax/df))+1)
    pp = np.zeros((vmax-vmin + 1,1), device = 'cuda')
    ll0 = np.zeros((np,nf))
    pp[:,0] = 1/np.linspace(vmin,vmax,vmax-vmin+1)
    ll0=1j*2*np.pi*df*np.mm(pp,x)
    mm=np.zeros((np,nf),dtype=np.complex64, device = 'cuda')
    abs_mm=np.zeros((np,nf), device = 'cuda')  
    l=np.zeros((np,offset),dtype=np.complex64, device = 'cuda')
    for luoj in range(lf-1,nf):
        mm[:,luoj]=np.mm(np.exp(ll0*(luoj)),d[:,luoj].view(-1,1)).squeeze()
    ml = np.abs(mm)[:,(lf-1):nf]
    return ml

def RTp_cudal(seismo_v1,ii,df,dt,np,vmin,vmax,fmin,fmax,a,b,dg,offset,m,M):
    ii = ii+1
    nend=M*ii-M+1-offset-m
    if nend<1:
        nend=0
    uxt=np.fliplr(seismo_v1[:,nend:(M*ii-M)-m+1]).T
    offset=M*ii-M+1-m-nend
    x=m+np.linspace(0,offset-1,offset)*dg
    x = x.view(1,-1).to('cuda')
    ccn=int(np.fix(np.tensor(1/df/dt)))
    d=np.fft.fft(uxt,n=ccn,dim=1)
    lf = int(np.round(np.tensor(fmin/df))+1)
    nf = int(np.round(np.tensor(fmax/df))+1)
    pp = np.zeros((vmax-vmin + 1,1), device = 'cuda')
    ll0 = np.zeros((np,nf))
    pp[:,0] = 1/np.linspace(vmin,vmax,vmax-vmin+1)
    ll0=1j*2*np.pi*df*np.mm(pp,x)
    mm=np.zeros((np,nf),dtype=np.complex64, device = 'cuda')
    abs_mm=np.zeros((np,nf), device = 'cuda')  
    l=np.zeros((np,offset),dtype=np.complex64, device = 'cuda')
    for luoj in range(lf-1,nf):
        mm[:,luoj]=np.mm(np.exp(ll0*(luoj)),d[:,luoj].view(-1,1)).squeeze()
    ml = np.abs(mm)[:,(lf-1):nf]
    return ml