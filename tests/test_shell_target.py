import sys, numpy as np; sys.path.insert(0,'scripts'); import scriptdirs; import shell_target as st
ok=True
def check(n,c,i=''): 
    global ok; ok&=bool(c); print(('PASS ' if c else 'FAIL ')+n, i)
W,C,sub=1.6,1.2,2; gx,gy=21,17; ii,jj=np.meshgrid(np.arange(gx),np.arange(gy)); x=ii*W/sub; y=jj*C/sub
# 1 flat sheet: lambda = 1
sh=dict(X=np.stack([x,y,0*x],-1),W=W,C=C,sub=sub)
P=np.c_[np.linspace(0.3,9,60),1+0.4*np.sin(np.linspace(0,9,60)),0.1*np.cos(np.linspace(0,20,60))]
check('flat sheet: stretch 1', np.allclose(st.segment_stretch(P,sh),1,atol=1e-12))
# 2 cylinder (isometric bend along x): lambda ~ 1 for in-plane yarn
r=4.0; cyl=np.stack([r*np.sin(x/r),y,r-r*np.cos(x/r)],-1); sh2=dict(sh,X=cyl)
Pf=P.copy(); Pf[:,2]=0; lam=st.segment_stretch(Pf,sh2); check('isometric cylinder: stretch 1 (chord error of a sampled analytic cylinder only, about (h/r)^2/24)', np.abs(lam-1).max()<3e-3, '%.2e'%np.abs(lam-1).max())
# 3 uniform stretch 1.1 in x: segments along x get 1.1
sh3=dict(sh,X=np.stack([1.1*x,y,0*x],-1)); Px=np.c_[np.linspace(1,8,30),np.full(30,3.0),np.zeros(30)]
check('uniform x stretch 1.1: course-direction segment 1.1', np.allclose(st.segment_stretch(Px,sh3),1.1))
# 4 conservation and sign
l0=np.random.default_rng(0).uniform(0.1,0.3,50); lam=1+0.05*np.random.default_rng(1).standard_normal(50)
f,eb=st.course_factors(l0,lam,-1,1); check('course total conserved', abs((l0*f).sum()-l0.sum())<1e-12*l0.sum())
check('sign -1: the most stretched segment gets the smallest factor', np.argmax(lam)==np.argmin(f))
f2,_=st.course_factors(l0,lam,1,1); check('sign +1 is the mirror image', np.allclose(f-1,-(f2-1)))
# 5 the real fold of a shell run: rigid triangles = folded mesh with all edges kept -> check that a pure rotation of the sheet gives exactly 1
th=0.7; R3=np.array([[1,0,0],[0,np.cos(th),-np.sin(th)],[0,np.sin(th),np.cos(th)]]); sh5=dict(sh,X=sh['X']@R3.T)
check('rigid rotation of the sheet: stretch exactly 1', np.allclose(st.segment_stretch(P,sh5),1,atol=1e-12))
print('ALL OK' if ok else 'SOME FAILED')
