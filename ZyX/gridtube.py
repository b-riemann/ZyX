# functions for the Grid-Cylinder Algorithm to find focusing terms
# B. Riemann, Oct 2020

from numpy import meshgrid, arange, sum, zeros, sin, cos, absolute, logical_and, hstack
from math import factorial
from scipy.linalg import lstsq
from pandas import DataFrame
from scipy.sparse import lil_matrix
from scipy.integrate import cumtrapz
from sympy import var, I, expand, Poly
from numpy import logical_and
from matplotlib.pyplot import subplots
from warnings import warn

def pqr_indices(maxOrder: int=6, rMaxOrder: int=0, ySymmetric: bool=True, dipole: bool=False) -> DataFrame:
    """
    select basis functions for potential in the cylinder"""
    pqr = list()
    for p in range(maxOrder+1):
        for q in range(maxOrder+1):
            for r in range(maxOrder+1):
                if p+q > maxOrder:
                    continue
                if q % 2 == 0 and ySymmetric: # y is an odd polynomial for machine-plane symmetry
                    continue
                if r > rMaxOrder:
                    continue
                if p+q == 0:
                    continue
                if p+q == 1 and not dipole:
                    continue
                pqr.append((p,q,r))
    df = DataFrame(columns=('p','q','r'), data=pqr)
    df['order'] = df['p'] + df['q']
    return df


def generateBasis(pqr: DataFrame, xf, yf, zf, dirs=(0,1), skalfac: float=1):
    """
    generate 2d basis functions for the indices specified by pqr,
    using flat (possibly unstructured) set of data points xf, yf, zf
    skalfac: 1 (m to m) or (mm to mm), 1000: (m to m)
    """
    nPts=len(xf)
    A = zeros((len(dirs)*nPts, len(pqr)), float) #later: empty
    for n, row in pqr.iterrows():
        # A[:, n] = xf**row['p'] * yf**row['q'] * zf**row['r']  # potential
        skal=skalfac**(row['order']-1) # mm to m for transverse coordinates / length unit ? check
        
        start=0
        for d in dirs:
            if d==0:
                if row['p'] > 0:
                    A[start:(start+nPts), n] = skal*row['p'] * xf**(row['p']-1) * yf**row['q'] * zf**row['r']  # Bx
            elif d==1:
                if row['q'] > 0:
                    A[start:(start+nPts):, n] = skal*row['q'] * xf**row['p'] * yf**(row['q']-1) * zf**row['r']  # By
            else:
                raise NotImplementedError('z dependence not yet ready')
            start+=nPts
    return A


def tubeSystemMatrix(traj_spos, triangle_width, twoDbasis, allS):
    """
    Note: this function uses all length inputs in [mm]
    traj_spos: s coordinates of trajectory
    triangle_width: scaling factor of the standard triangle function - it will reach zero at this argument value
    twoDbasis: a matrix constructed by generateBasis
    allS: s coordinates of all grid points in the tube system
    """
    bigA=zeros((twoDbasis.shape[0], twoDbasis.shape[1]*len(traj_spos)))
    dblS = hstack((allS,allS)) # as the right side of the system is a hstack of bX, bY
    colstart = 0
    for spos in traj_spos[:-1]:
        difs = absolute(dblS-spos)
        msk =  difs < triangle_width
        bigA[msk, colstart:colstart+twoDbasis.shape[1]] = twoDbasis[msk] * (1 - difs[msk,None] / triangle_width) # triangle function
        colstart += twoDbasis.shape[1]
    difs = dblS - traj_spos[-1]
    msk = logical_and(difs > -triangle_width, difs<=0)
    bigA[msk, colstart:colstart+twoDbasis.shape[1]] = twoDbasis[msk] * (1 + difs[msk,None] / triangle_width) # triangle left side
    msk = difs>0
    bigA[msk, colstart:colstart+twoDbasis.shape[1]] = twoDbasis[msk] # box right side
    return bigA


def ra_to_rr(xa,za,theta):
    ct=cos(theta)
    st=sin(theta)
    xr = ct*xa + st*za
    zr = -st*xa + ct*za    
    return xr, zr

def regionpoints(xc: float, yc: float, zc: float, xin, yin, zin, radius: float, theta: float, h: float, nopts_warn: bool=True):
    """
    return all points (flattened) that are in a region with given radius parameter
    if h is positive, region is a cylinder with height = 2*h
    else, region is a spheroid with transverse semiaxes as radius and rotational semiaxis -h.
    """
    xr, zr = ra_to_rr(xin-xc, zin-zc, theta)
    yr = yin-yc    

    if h < 0: # spheroid
        inRegion = (xr**2 + yr**2)/radius**2 + (zr/h)**2 < 1        
    else: # cylinder
        inRegion = logical_and(xr**2 + yr**2 < radius**2, absolute(zr) < h)
        
    if nopts_warn:
        if sum(inRegion) == 0:
            raise Exception('no points in region!')
    return inRegion, xr[inRegion].flatten(), yr[inRegion].flatten(), zr[inRegion].flatten()


def genB(bInRegion, theta=0, curv=0):
    """
    generate b Vector in correct shape for linear equation system,
    based on the full B array return by read_operabox, then masked with the region mask
    theta: 
    curv: curvature to be substracted from bY
    """
    # bFull[inRegion]
    bX, bZ = ra_to_rr(bInRegion[:,0], bInRegion[:,2], theta)
    # bZ is not used (yet)
    bY = bInRegion[:,1] - curv
    return bX, bY #hstack((bX,bY))


def solver(A, b, pqr: DataFrame, skalfac: float=1):
    # global static: pqr
    coefs, resid, rank, s = lstsq(A, b) # coefs in units of cm
    for n, row in pqr.iterrows():
        coefs[n] *= skalfac**(row['order']-1)
    return coefs, resid, s[0]/s[-1]  #  scaled coefs, fit residual, condition number


def fitprocedure(xyzCenter_mm: iter, xyzIn_mm, pqr: DataFrame, bFull, theta: float, curv: float, radius: float=9.5, h: float=0.55):
    # global static: xyzIn, pqr, bFull
    inRegion, xf, yf, zf = regionpoints(*xyzCenter_mm, *xyzIn_mm, radius, theta, h)
    #print(f'{len(xf) = }')
    A = generateBasis(pqr, xf, yf, zf, skalfac=0.1) # gradients in units of cm, yields superior condition numbers < 100 to due radius being in the order of 1 cm # skalfac 0.1
    #print(f'{A.shape = }')
    bf = genB(bFull[inRegion], theta, curv)
    #print(f'{bf.shape = }')
    return *solver(A, hstack(bf), pqr, skalfac=100), len(xf) # coefs [m], fit residual, condition number, n_points  # skalfac 100


def plot_multipole_order(ax, s: iter, coefs: iter, pq: DataFrame, order: int, basis: str=r'$\Psi$', integrated: bool=False, legend: bool=True, numvalues: bool=True ,**kwargs):
    found = False
    for n, row in pq.iterrows():
        if row['order']!=order:
            continue
        found = True
        p,q = [row[c] for c in ('p', 'q')]
        if integrated:
            cu = cumtrapz(coefs[n], s)
            lbl = (r'int.%s[%i,%i]=%.5e' % (basis,p,q,cu[-1])) if numvalues else r'int.%s[%i,%i]' % (basis,p,q)
            ax.plot(s[1:], cu, label=lbl if legend else None, **kwargs)
        else:
            ax.plot(s, coefs[n], label=(r'%s[%i,%i]' % (basis,p,q)) if legend else None, **kwargs)
    # following lines just prevent a corner case that looks weird in a plot:
    # if 0 is not displayed on the y axis, we want it to be displayed
    yl = ax.get_ylim()
    if yl[0]*yl[1] > 0:
        if yl[0] > 0:
            ax.set_ylim((0, yl[1]*1.05))
        else:
            ax.set_ylim((yl[0]*1.05, 0))
    if legend and found:
        ax.legend()


def plot_multipoles(s: iter, coefs: iter,
                    pq: DataFrame,
                    basis: str=r'$\Psi$',
                    integrated: bool=False,
                    axo=None,
                    legend=True, numvalues=True,
                    **kwargs):
    # global static: traj
    newfig = axo is None
    if newfig:
        fig, axo = subplots(2,3,sharex=True,figsize=(15,8))

    maxorder = min(axo.size, pq['order'].max())  # must be python min, import numpy amin instead of numpy min if you want to use it
    for order in range(1,maxorder+1):
        try:
            ax=axo.flat[order-1]
        except:
            raise Exception('Not enough axes for maximum pq order. use more subplots or manipulate pq')
        plot_multipole_order(ax, s, coefs, pq, order, basis, integrated, legend, numvalues, **kwargs)

    if newfig:
        return fig, axo


def zzc_coefmatrix(p: int, q: int, realPart: bool):
    ## return matrix with coefficients mat[xorder, yorder]
    x,y = var('x,y', real=True)
    f = (x+I*y)**p * (x-I*y)**q
    # now find coefficients of a polynomial
    fx=Poly(f,x)
    mat=zeros((p+q+1,p+q+1), dtype=complex)
    for r, g in enumerate(fx.all_coeffs()):
        fg=Poly(g,y)
        ac=fg.all_coeffs()
        mat[-r-1,:len(ac)] = ac[::-1]
    #print(mat)
    #print(expand(f))
    return lil_matrix(mat.real if realPart else mat.imag)


def zzc_indices(maxOrder: int, ySymmetric: bool=True, dipole: bool=False, pure: bool=False) -> DataFrame:
    zzc = list()
    for p in range(maxOrder+1):
        for q in range(maxOrder+1):
            if q>p: # conjugate solution
                continue
            if q==p and ySymmetric: # rotational symmetry
                continue
            if q!=0 and pure: # filter only 'classic' multipoles that fulfill 2d Laplace
                continue
            if p+q > maxOrder:
                continue
            if p+q == 0:
                continue
            if p+q == 1 and not dipole:
                continue  
            for re in (True,False):
                if re and ySymmetric:
                    continue
                zzc.append((p,q,re))
    df = DataFrame(columns=('z', 'zc', 'real'), data=zzc)     
    df['order'] = df['z'] + df['zc']
    return df


def to_generalized_multipoles(pqr: DataFrame, normalization=None):
    """
    normalization:
      'pangaea': pure potential Re,Im(z..z*..)
      'europe': power series in B_r=d Pot / dr, yields n in divisor 
                "k_{n-1}" = (d^{n-1} B_r / d_r^{n-1}) / {n-1}! = d^n Pot / d_r^n / n!,
                used in opa, tracy, most of Europe
      'america': n-th derivative in B, "k_n" = (d B_r / d_r^n),
                  used by mad, madx, and presumeably elegant
      This also fits perfectly with Pascals triangle, as the coefficient in front of x^{n-1} y is +n.
      
      --CONSIDERATONS on the multipole strength in 'european' and 'american' notation--
      * the 'plus'-potential (no minus sign, e.g. B_x = d \Psi_x / d x) of a 'standard' upright 2n-pole fulfilling 2d laplace is proportional Im (x+iy)^n
      -> it can be shown that the 'american coefficient' for this expression is
             coef_american = d^{n-1} B_y / d x^{n-1} = d^{n-1} / d x{n-1} * d / dy Psi = n!
         therefore the 'american' multipole, scaling above expression to 1, is
             Psi_american = Im (z^n) / n!
      -> the european notation is the coefficient of the power series in B_y, the Taylor coefficient
             coef_european = 1/(n-1)! d^{n-1} B_y / d x^{n-1} = coef_american / (n-1)! = n
         therefore the 'european' multipole, scaling above expression to 1, is
             Psi_european = Im (z^n) / n
    """
    norms = ('pangaea', 'europe', 'america')
    if normalization not in norms:
        raise NotImplementedError(f'select normalization from {norms}')
    # no checks are performed if all elements are filled..
    maxOrder = pqr['order'].max()
    zzc = zzc_indices(maxOrder)
    
    coefmat = zeros((len(zzc), len(pqr)))
    for n, row in zzc.iterrows():
        mats = zzc_coefmatrix(row['z'], row['zc'], row['real']) # scipy.sparse.lil_matrix    
        for p, q in zip(*mats.nonzero()):
            #print(p,q)
            m = pqr.loc[logical_and(pqr['p']==p,pqr['q']==q)].index[0]
            coefmat[n,m] = mats[p,q]
        if normalization=='america':
            coefmat[n] /= factorial( row['order'] ) 
        elif normalization=='europe':
            coefmat[n] /= row['order']
        # 'pangaea' doesn't change the norm
    return zzc, coefmat
