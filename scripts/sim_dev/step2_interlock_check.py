import numpy as np
from scipy.interpolate import splprep, splev
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def loop_points(c, w, d, L, x_offset=0.0):
    """9 control points P1..P9 for one jersey loop, per Ravandi et al. 2022.
    x_offset shifts the whole loop along the course direction."""
    P1 = [0 + x_offset,           0,            d/2]
    P2 = [(c+2*d)/4 + x_offset,   (L-w)/2,      0]
    P3 = [c/4 + x_offset,         L/2,         -d/2]
    P4 = [(c-2*d)/4 + x_offset,   (L+w)/2,      0]
    P5 = [c/2 + x_offset,         L,            d/2]
    P6 = [(3*c+2*d)/4 + x_offset, (L+w)/2,      0]
    P7 = [3*c/4 + x_offset,       L/2,         -d/2]
    P8 = [(3*c-2*d)/4 + x_offset, (L-w)/2,      0]
    P9 = [c + x_offset,           0,            d/2]
    return np.array([P1,P2,P3,P4,P5,P6,P7,P8,P9])

def fit_spline(points, k=3):
    tck, u = splprep(points.T, s=0, k=k)
    u_fine = np.linspace(0, 1, 200)
    x, y, z = splev(u_fine, tck)
    return x, y, z

if __name__ == "__main__":
    c, w, d, L = 4.0, 3.0, 0.5, 5.0

    loop1 = loop_points(c, w, d, L, x_offset=0.0)
    loop2 = loop_points(c, w, d, L, x_offset=c)   # shifted by course spacing c

    # THE CHECK: loop2's P1 must equal loop1's P9
    p9_loop1 = loop1[8]
    p1_loop2 = loop2[0]
    match = np.allclose(p9_loop1, p1_loop2)
    print("loop1 P9:", p9_loop1)
    print("loop2 P1:", p1_loop2)
    print("MATCH (should be True):", match)

    x1,y1,z1 = fit_spline(loop1)
    x2,y2,z2 = fit_spline(loop2)

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x1,y1,z1,'r-',lw=2,label='loop 1')
    ax.plot(x2,y2,z2,'b-',lw=2,label='loop 2 (shifted by c)')
    ax.scatter(loop1[:,0],loop1[:,1],loop1[:,2],c='k',s=20)
    ax.scatter(loop2[:,0],loop2[:,1],loop2[:,2],c='gray',s=20)
    ax.set_xlabel('x (course)')
    ax.set_ylabel('y (wale/height)')
    ax.set_zlabel('z (thickness)')
    ax.legend()
    plt.tight_layout()
    plt.savefig('step2_interlock_check.png', dpi=150)
    print("Saved step2_interlock_check.png")
