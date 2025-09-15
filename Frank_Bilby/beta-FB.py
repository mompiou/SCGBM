import numpy as np
from matplotlib import pyplot as plt


def null(A, rcond=None):

    u, s, vh = np.linalg.svd(A, full_matrices=True)
    M, N = u.shape[0], vh.shape[1]
    if rcond is None:
        rcond = np.finfo(s.dtype).eps * max(M, N)
    tol = np.amax(s) * rcond
    num = np.sum(s > tol, dtype=int)
    Q = vh[num:, :].T.conj()
    return Q
    
    
def Sy():
        S1 = Rot(90, 1, 0, 0)
        S2 = Rot(180, 1, 0, 0)
        S3 = Rot(270, 1, 0, 0)
        S4 = Rot(90, 0, 1, 0)
        S5 = Rot(180, 0, 1, 0)
        S6 = Rot(270, 0, 1, 0)
        S7 = Rot(90, 0, 0, 1)
        S8 = Rot(180, 0, 0, 1)
        S9 = Rot(270, 0, 0, 1)
        S10 = Rot(180, 1, 1, 0)
        S11 = Rot(180, 1, 0, 1)
        S12 = Rot(180, 0, 1, 1)
        S13 = Rot(180, -1, 1, 0)
        S14 = Rot(180, -1, 0, 1)
        S15 = Rot(180, 0, -1, 1)
        S16 = Rot(120, 1, 1, 1)
        S17 = Rot(240, 1, 1, 1)
        S18 = Rot(120, -1, 1, 1)
        S19 = Rot(240, -1, 1, 1)
        S20 = Rot(120, 1, -1, 1)
        S21 = Rot(240, 1, -1, 1)
        S22 = Rot(120, 1, 1, -1)
        S23 = Rot(240, 1, 1, -1)
        S24 = np.eye(3, 3)
        S=[S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11, S12, S13, S14, S15, S16, S17, S18, S19, S20, S21, S22, S23, S24]
        return S

def Rot(th, a, b, c):
    th = th * np.pi / 180
    no = np.linalg.norm([a, b, c])
    aa = a / no
    bb = b / no
    cc = c / no
    c1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], float)
    c2 = np.array([[aa**2, aa * bb, aa * cc], [bb * aa, bb**2, bb * cc], [cc * aa,
                                                                          cc * bb, cc**2]], float)
    c3 = np.array([[0, -cc, bb], [cc, 0, -aa], [-bb, aa, 0]], float)
    R = np.cos(th) * c1 + (1 - np.cos(th)) * c2 + np.sin(th) * c3

    return R

def angle(A):
        if 0.5 * (np.trace(A + np.eye(3)) - 1) > 1:
            return 0
        elif 0.5 * (np.trace(A + np.eye(3)) - 1) < -1:
            return 180
        else:
            return np.arccos(0.5 * (np.trace(A + np.eye(3)) - 1)) * 180 / np.pi

axes=[[0,0,1],[1,1,1],[2,3,4]]
plan=[[1,0,0],[-1,1,0],[0,4,-3]]
S=Sy()
off=0
for u,n in zip(axes,plan):
    u=u/np.linalg.norm(u)
    n=n/np.linalg.norm(n)
    p=np.cross(u,n)
    p=p/np.linalg.norm(p)

    coupling=[]
    color_min=[]
    ang=np.arange(1,90,0.5)

    for th in ang:
        b=[]
        co=[]
        for G1 in S:
                R=np.dot(G1,Rot(th,u[0],u[1],u[2]).T)-np.eye(3,3)
                v = null(R, 0.001)
                if np.shape(v)[1]>0:
                    theta=angle(R)
                    t=np.dot(n,v)[0]
                    B=np.dot(R,p)
                    if np.abs(t)<0.0001 and np.linalg.norm(B)>1e-7:
                        n2=B/np.linalg.norm(B)
                        beta=np.linalg.norm(B)/np.dot(n2,n)
                        b.append(np.abs(beta))
                        co.append(np.abs(t))
                        #print(beta,B,v.T,theta,t)
        coupling.append(np.min(b))
        mi=np.argwhere(np.abs(b-np.min(b))<1e-5)
        cc=[]
        for i in range(0,mi.shape[0]):
            cc.append(co[mi[i][0]])

        color_min.append(mi[np.argmin(cc)][0])
        
        

#    plt.scatter(ang-off,coupling, s=2)
    plt.plot(ang-off,coupling,'-',linewidth=2)
    off=off+1
    #plt.scatter(ang,coupling,c=color_min)
    #plt.colorbar()
#plt.plot(ang,2*np.tan(ang*np.pi/180/2))
plt.rcParams['text.usetex'] = True
plt.legend([r'$\langle 001 \rangle$', r'$\langle 111 \rangle$','others'], loc="upper left")
plt.ylabel(r'coupling factor $\beta$')
plt.xlabel(r'angle ($^\circ$)')
plt.show()


