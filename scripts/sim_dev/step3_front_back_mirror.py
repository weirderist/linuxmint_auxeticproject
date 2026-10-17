import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def loop_points(c, w, d, L, x_offset=0.0, mirror=False):
    """9 control points for one jersey loop.
    mirror=True gives the back-loop version (front loop viewed from other side)."""
    P1 = [0,            0,            d/2]
    P2 = [(c+2*d)/4,    (L-w)/2,      0]
    P3 = [c/4,          L/2,         -d/2]
    P4 = [(c-2*d)/4,    (L+w)/2,      0]
    P5 = [c/2,          L,            d/2]
    P6 = [(3*c+2*d)/4,  (L+w)/2,      0]
    P7 = [3*c/4,        L/2,         -d/2]
    P8 = [(3*c-2*d)/4,  (L-w)/2,      0]
    P9 = [c,            0,            d/2]
    pts = np.array([P1,P2,P3,P4,P5,P6,P7,P8,P9])

    if mirror:
        pts[:,2] = -pts[:,2]   # flip z (thickness) -> back loop

    pts[:,0] += x_offset
    return pts

def fit_spline(points, k=3):
    tck, u = splprep(points.T, s=0, k=k)
    u_fine = np.linspace(0, 1, 200)
    x, y, z = splev(u_fine, tck)
    return x, y, z

if __name__ == "__main__":
    c, w, d, L = 4.0, 3.0, 0.5, 5.0

    front = loop_points(c, w, d, L, x_offset=0.0, mirror=False)
    back  = loop_points(c, w, d, L, x_offset=0.0, mirror=True)

    print("front P1,P5,P9 z-values:", front[0,2], front[4,2], front[8,2])
    print("back  P1,P5,P9 z-values:", back[0,2],  back[4,2],  back[8,2])

    xf,yf,zf = fit_spline(front)
    xb,yb,zb = fit_spline(back)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(xf,yf,zf,'r-',lw=2,label='front loop')
    ax.plot(xb,yb,zb,'g-',lw=2,label='back loop (mirrored)')
    ax.scatter(front[:,0],front[:,1],front[:,2],c='darkred',s=20)
    ax.scatter(back[:,0],back[:,1],back[:,2],c='darkgreen',s=20)
    ax.set_xlabel('x (course)')
    ax.set_ylabel('y (wale/height)')
    ax.set_zlabel('z (thickness)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('step3_front_back_mirror.png', dpi=150)
    print("Saved step3_front_back_mirror.png")
