"""Reproduce the figures and numerical checks in mband_wavelets.tex."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from scipy.signal import upfirdn

OUT = Path(__file__).resolve().parent
F = OUT / "figures"
F.mkdir(parents=True, exist_ok=True)
H3 = np.array([
    [0.33838609728386, 0.53083618701374, 0.72328627674361,
     0.23896417190576, 0.04651408217589, -0.14593600755399],
    [-0.11737701613483, 0.54433105395181, -0.01870574735313,
     -0.69911956479289, -0.13608276348796, 0.42695403781698],
    [0.40363686892892, -0.62853936105471, 0.46060475252131,
     -0.40363686892892, -0.07856742013185, 0.24650202866523],
])
r3 = np.sqrt(3)
a2 = np.array([1+r3, 3+r3, 3-r3, 1-r3]) / (4*np.sqrt(2))
b2 = (-1.) ** np.arange(4) * a2[::-1]
H2 = np.array([a2, b2])

def mlt(M):
    n = np.arange(2*M)
    w = np.sin(np.pi*(n+.5)/(2*M))
    k = np.arange(M)[:,None]
    return w * np.sqrt(2/M) * np.cos(np.pi/M*(k+.5)*(n+.5+M/2.))

H4 = mlt(4)

def fold(y, N):
    out = np.zeros(N, dtype=np.result_type(y, 1.0))
    np.add.at(out, np.arange(len(y)) % N, y)
    return out

def analysis(x, H):
    M, N = len(H), len(x)
    assert N % M == 0
    return [fold(upfirdn(h, x, 1, M), N//M) for h in H]

def synthesis(cs, H):
    M = len(H)
    N = M * len(cs[0])
    L = H.shape[1]
    parts = [upfirdn(h[::-1], c, M, 1) for h, c in zip(H, cs)]
    raw = np.zeros(max(map(len,parts)))
    for part in parts:
        raw[:len(part)] += part
    return np.roll(fold(raw, N), -(L-1))

def multilevel(x, H, J):
    approx = x.copy()
    details = []
    for _ in range(J):
        cs = analysis(approx, H)
        approx = cs[0]
        details.append(cs[1:])
    return approx, details

def multirec(approx, details, H):
    y = approx.copy()
    for layer in reversed(details):
        y = synthesis([y] + layer, H)
    return y

def pr_residual(H):
    M, L = H.shape
    G = H[:, ::-1]
    q = int(np.ceil(L/M))
    E = np.zeros((M,M,q))
    R = np.zeros((M,M,q))
    for k in range(M):
        for r in range(M):
            e = H[k,r::M]
            g = G[k,M-1-r::M]
            E[k,r,:len(e)] = e
            R[r,k,:len(g)] = g
    P = np.zeros((M,M,2*q-1))
    for i in range(M):
        for j in range(M):
            for k in range(M):
                P[i,j] += np.convolve(R[i,k], E[k,j])
    diag = np.array([P[i,i] for i in range(M)]).mean(axis=0)
    n0 = int(np.argmax(np.abs(diag)))
    gain = diag[n0]
    target = np.zeros_like(P)
    for i in range(M):
        target[i,i,n0] = gain
    return np.max(abs(P-target))/abs(gain), gain, M*n0+M-1

rng = np.random.default_rng(2026)
results = []
for name,H in [('D4 / db2', H2), ('Lin et al. 3-band',H3), ('MLT 4-band',H4)]:
    M, N = len(H),len(H)**5
    x = rng.standard_normal(N)
    y = synthesis(analysis(x,H),H)
    cs = analysis(x,H)
    e = np.linalg.norm(x-y)/np.linalg.norm(x)
    p,g,d = pr_residual(H)
    energy = abs(np.dot(x,x)-sum(np.dot(b,b) for b in cs))/np.dot(x,x)
    results.append(dict(name=name,M=M,L=H.shape[1],relative_reconstruction_error=float(e),
       relative_energy_error=float(energy),polyphase_residual=float(p),gain=float(g),delay=int(d)))
(OUT / "metrics.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))

fig, axs = plt.subplots(1,2,figsize=(11.4,3.3),layout="constrained")
for ax,name,H in zip(axs,['3-band, 2-moment filters (Lin et al.)','4-band MLT, length 8'],[H3,H4]):
    w = np.linspace(0,np.pi,700)
    for k,h in enumerate(H):
        z = np.exp(-1j*np.outer(w,np.arange(len(h))))@h
        ax.plot(w/np.pi,20*np.log10(np.maximum(abs(z),1e-4)),label=f'channel {k}',lw=1.6)
    ax.set_title(name,fontsize=10)
    ax.set_xlim(0,1)
    ax.set_ylim(-80,15)
    ax.set_xlabel(r'normalized frequency $\omega/\pi$')
    ax.grid(alpha=.22)
    ax.legend(fontsize=8,ncol=2,loc='lower right')
axs[0].set_ylabel('analysis-filter magnitude (dB)')
fig.savefig(F/'frequency_responses.pdf',bbox_inches='tight')
plt.close(fig)

N = 729
t = np.arange(N)/N
x = np.sin(2*np.pi*5*t)+.5*(t>.6)+.35*np.exp(-.5*((t-.31)/.011)**2)+.05*np.random.default_rng(3).standard_normal(N)
A,ds = multilevel(x,H3,4)
xr = multirec(A,ds,H3)
Ee = [float(np.sum(A*A))]+[float(np.sum(a*a)) for layer in ds for a in layer]
print('multilevel relative error',np.linalg.norm(x-xr)/np.linalg.norm(x),
      'energy rel err',abs(np.dot(x,x)-sum(Ee))/np.dot(x,x))
fig,axs = plt.subplots(5,1,figsize=(10,7.5),
        gridspec_kw={'height_ratios':[1.45,1,1,1,1]},layout='constrained')
axs[0].plot(t,x,lw=1)
axs[0].axvline(.6,ls=':',lw=.8,c='0.5')
axs[0].set_ylabel('signal')
for j in range(4):
    for k,d in enumerate(ds[j]):
        grid = np.arange(len(d))*3**(j+1)/N
        axs[j+1].plot(grid,d,lw=.85,label=f'$D_{{{j+1},{k+1}}}$')
    axs[j+1].axvline(.6,ls=':',lw=.8,c='0.5')
    axs[j+1].set_ylabel(f'level {j+1}')
    axs[j+1].legend(loc='upper right',fontsize=7,ncol=2)
for ax in axs:
    ax.set_xlim(0,1)
    ax.grid(alpha=.18)
axs[-1].set_xlabel('normalized time')
fig.savefig(F/'three_band_multilevel.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax = plt.subplots(figsize=(7.0,2.55))
ax.set_xlim(-0.2,7.6)
ax.set_ylim(-2.55,0.25)
ax.axis('off')
positions={'x':(2.35,0),'a1':(1.25,-1.1),'d11':(3.15,-1.1),'d12':(5.05,-1.1),
           'a2':(0.3,-2.2),'d21':(1.65,-2.2),'d22':(3.0,-2.2)}
labels={'x':r'$x$', 'a1':r'$A_1$', 'd11':r'$D_{1,1}$','d12':r'$D_{1,2}$',
        'a2':r'$A_2$','d21':r'$D_{2,1}$','d22':r'$D_{2,2}$'}
for name, pos in positions.items():
    ax.text(pos[0],pos[1],labels[name],ha='center',va='center',fontsize=13,
        bbox=dict(boxstyle='square,pad=0.25',fc='white',ec='black',lw=.7))
for src,dst in [('x','a1'),('x','d11'),('x','d12'),('a1','a2'),('a1','d21'),('a1','d22')]:
    pa,pb=positions[src],positions[dst]
    vec=(pb[0]-pa[0],pb[1]-pa[1])
    norm=np.hypot(*vec)
    start=(pa[0]+vec[0]*.22/norm,pa[1]+vec[1]*.22/norm)
    end=(pb[0]-vec[0]*.27/norm,pb[1]-vec[1]*.27/norm)
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='->',mutation_scale=10,linewidth=.85,color='black'))
ax.text(5.4,-2.12,r'$M=3$',fontsize=12,ha='left')
fig.savefig(F/'multiresolution_tree.pdf',bbox_inches='tight',pad_inches=.1)
plt.close(fig)
