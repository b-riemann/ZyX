"""
ZX plane (machine plane) motion of (reference) particles,
computation of focusing terms using Automatic Differentiation.

Bernard Riemann (2019/2020)
"""
from autograd import elementwise_grad
from numpy import absolute, allclose, amin, amax, asarray, zeros, arange, linspace, \
                  zeros_like, pi, ones_like, mean, diff, floor, empty, array, concatenate, \
                  sin, cos, meshgrid, trapz, copy, logical_and, sum, argmin, argmax, Inf
from scipy.integrate import cumtrapz
from scipy.special import factorial
from scipy.optimize import root_scalar, minimize_scalar
from pandas import DataFrame, Series

from .analytic import LinearEdgeMagnet, GammaEdgeMagnet
from .basic import AbstractField, rigidityFromMeV

def df_print(df: DataFrame, filename: str):
    #stri = df.__str__()
    #strinew = '#' + stri[1:]  # strings are immutable in python
    #with open(filename, 'w') as f:
    #    f.write(strinew)
    df.to_csv(filename, sep=' ', index=False)

def clusterDipoleQuad(term, ds, interleave=5):
    sumOver = 2*interleave
    regularElems = int(floor(term.size/sumOver))
    regularPoints = regularElems*sumOver
    if term.size > regularPoints:
        integratedStrength = empty(regularElems+1, float)
        integratedStrength[regularElems] = ds*trapz(term[(regularPoints-1):])
        dL = ds*sumOver*ones_like(integratedStrength)
        dL[-1] = ds*(term.size-regularPoints)
    else:
        integratedStrength = empty(regularElems, float)
        dL = ds*sumOver*ones_like(integratedStrength)
    x = empty((regularElems, sumOver+1))
    x[:,:-1] = term[:regularPoints].reshape((regularElems, sumOver))
    x[:,-1] = term[sumOver:(regularPoints+1):sumOver]
    integratedStrength[:regularElems] = ds*trapz(x, axis=1)
    return integratedStrength, dL


def clusterNonlinear(term, ds, interleave=5):
    sumOver = 2*interleave
    regularElems = int(floor((term.size-interleave)/sumOver))
    regularPoints = interleave + regularElems*sumOver
    if term.size > regularPoints:
        integratedStrength = empty(regularElems+2, float)
        integratedStrength[regularElems+1] = ds*trapz(term[(regularPoints-1):])
    else:
        integratedStrength = empty(regularElems+1, float)
    integratedStrength[0] = ds*trapz(term[:(interleave+1)])
    x = empty((regularElems, sumOver+1))
    x[:,:-1] = term[interleave:regularPoints].reshape((regularElems, sumOver))
    x[:,-1] = term[(interleave+sumOver):(regularPoints+1):sumOver]
    integratedStrength[1:(regularElems+1)] = ds*trapz(x, axis=1)
    return integratedStrength


class FocusingTerms():
    def __init__(self, functionZXn, trajectory, minOrder=0, maxOrder=3, name='', rigidity=None):
        # def buildGradients(fun, maxOrder):
        funList = [functionZXn]
        for p in range(1, maxOrder + 1):
            funList.append(elementwise_grad(funList[-1], 4))  # 4: derivative regarding beamx parameter

        self.terms = zeros((maxOrder + 1, len(trajectory)), float)
        arZero = zeros(len(trajectory), float)
        for p in range(minOrder, maxOrder + 1):
            # print(p)
            self.terms[p] = funList[p](trajectory.Z, trajectory.X, trajectory.cThetaSin, trajectory.cThetaCos, arZero)
            self.terms[p] /= factorial(p)  # 'American' to 'European' notation
        self.s = trajectory.s
        self.name = name
        self.rigidity = rigidity

    def toFrame(self, df: DataFrame):
        assert df['s'].equals(Series(self.s))
        for n in range(self.terms.shape[0]):
            term = copy(self.terms[n])
            term *= self.s[1]-self.s[0] # ds
            term[0] /= 2
            df[f'{self.name}{n}int'] = term
        return df

    def kickDriftExport(self, filename, crunchAfterSlice: int=-1, startSymmetric: bool=True):
        # for tracy-null
        tT = copy(self.terms.T)
        # in kick-drift-kick, end kicks should have half-length due to symmetry:
        # tT[0]  /= 2
        # tT[-1] /= 2

        hRef = copy(tT[:,0])  # separate reference trajectory from remaining kick
        tT[:,0] = 0

        sDif = diff(self.s)
        dL = mean(sDif)
        # we assume: assert dL_kick == sDif for all elements

        with open(filename, 'w') as f:
            def _writeKick(hRefi, tSlice, leni):
                f.write("\nkick len %.8e hRef %.8e mpa" % (leni, hRefi))
                f.write((" %.8e" * (tT.shape[1])) % tuple(tSlice))

            f.write('# kick-drift export (upright multipoles) for tracy-null')
            f.write('\norder %i' % tT.shape[1])
            _writeKick(hRef[0], tT[0], dL/2 if startSymmetric else dL)

            inner_elems: int= len(self.s)-1 if crunchAfterSlice < 0 else crunchAfterSlice
            for n in range(1,inner_elems):
                f.write("\ndrif %.8e" % dL)
                _writeKick(hRef[n], tT[n], dL)
            f.write("\ndrif %.8e" % dL)
            if crunchAfterSlice < 0:
                _writeKick(hRef[inner_elems], tT[inner_elems], dL/2)
            else:
                hCrunch = sum(hRef[inner_elems:])
                tCrunch = sum(tT[inner_elems:], axis=0)
                _writeKick(hCrunch, tCrunch, dL)

        print(f'KickDrift sequence exported to {filename}')

    def integrate(self, order, sStart=0, sEnd=Inf):
        """integrate focusing terms of given order in the interval s in [sStart,sEnd]"""
        msk = logical_and(self.s >= sStart, self.s <= sEnd)
        return trapz(self.terms[order][msk], self.s[msk])

    def integrated(self, pRange=None):
        if pRange is None:
            return cumtrapz(self.terms, self.s, axis=1, initial=0)
        else:
            return [cumtrapz(self.terms[p], self.s, initial=0) for p in pRange]

    def __getitem__(self, p):
        return self.terms[p]

    def __len__(self):
        return self.terms.shape[0]

    def _plotBase(self, ax, refRadius, x, stri, pRange=None, rigid=True):
        mcolors = ('black', 'red', 'lightgreen', 'magenta')
        mlinewidths = (1.3, 1.3, 1.1)
        # ax.set_title(r'$R = %i$ mm' % (refRadius * 1000))
        plotRigid = rigid and self.rigidity is not None
        preFactor = self.rigidity if plotRigid else 1
        if pRange is None:
            pRange = arange(len(self))
        pow = preFactor * (refRadius ** pRange)
        for n, p in enumerate(pRange):
            ax.plot(self.s,  x[n] * pow[n], label=stri.format(p=p, name=self.name),
                    linewidth=mlinewidths[p] if p<3 else 0.7, color=mcolors[p] if p<4 else f'C{p}', zorder=pRange[-1]+1-p)
        ax.set(xlabel='s [m]', ylabel='B [T]' if plotRigid else 'multipole strength', xlim=(self.s[0], self.s[-1]))
        # ax.legend()

    def plotRefRadius(self, ax, refRadius=0.010, rigid=True):
        self._plotBase(ax, refRadius, self, r'$R^{p} {name}_{p}$', rigid=rigid)
        [ax.spines[dir].set_color(None) for dir in ('top', 'right')]

    def plotIntegrated(self, ax, pRange=None, refRadius=0.010):
        self._plotBase(ax, refRadius, self.integrated(pRange), r'$R^{p} \int {name}_{p}\;ds$', pRange=pRange,
                       rigid=False)
        ax.set_ylabel(r'$\int b(s) \; ds$')

    def plot(self, ax, order: int):
        ax.plot(self.s, self.terms[order])
        ax.set(xlabel='s [m]', ylabel=f'd^{order} b_y / d x^{order}')


class PolyPotential:
    """holds mixed derivatives along (beamx, beamy), up to a given order.

    can be seen as (also cleaned-up) extension of FocusingTerms that incorporates everything, by breaking the assumption that the transverse magnetic field perpendicular to the trajectory consists of pure multipoles, but is a general polynomial in field components.

    requires 3d fields to be used properly"""
    def __init__(self, traj, cf, maxOrder=3):
        """bYmap should be instance of YExtendedSplineMap"""
        #assert True #that fields are all normalized by ridigity

        def potAroundTraj(beamx, beamy):
            # this is effectively only b_transx for bYmap
            # field in beamy === y direction
            # fld_y === bYmap
            return cf.potentialOnPath(traj.X + traj.cThetaCos * beamx, beamy, traj.Z + traj.cThetaSin * beamx)

        def derivs_beam(funct, direct='x'):
            funlist = [funct]
            for p in range(maxOrder):
                funlist.append(elementwise_grad(funlist[-1], 0 if direct=='x' else 1))
            return funlist

        # this creates a two-level list
        dxy_func = [derivs_beam(cfun, direct='y') for cfun in derivs_beam(potAroundTraj, direct='x')]
        # dont forget factorial prefactor

        # now: for each curv_x, curv_y, compute mixed derivatives in beamx, beamy
        self.dxy = zeros((maxOrder+1,maxOrder+1,len(traj)))

        xOrder = arange(maxOrder+1)
        yOrder = arange(maxOrder+1)
        self.maxCompleteOrder = min(xOrder[-1], yOrder[-1])  # built-in python max

        arZero = zeros(len(traj))
        for p in xOrder:
            for q in yOrder:
                fprod = factorial(p)*factorial(q)  # 'American' to 'European' notation
                self.dxy[p,q] = dxy_func[p][q](arZero, arZero) / fprod

        self.s = traj.s
        self.startSymmetric = traj.startSymmetric

        sDif = diff(self.s)
        self.dL = mean(sDif)
        assert allclose(self.dL, sDif)

    def getSlice(self, n: int, half: bool=False, nEnd: int=0):
        dL: float = self.dL/2 if half else self.dL
        multcoefs = list()
        # this loop will only iterate over complete polynomial orders

        if nEnd==0: # formely crunch=False
            dxy = self.dxy[:,:,n]
        elif nEnd<0: #formerly crunch=True
            dxy = sum( self.dxy[:,:,n:], axis=2 )
        else:
            dxy = sum( self.dxy[:,:,n:(nEnd+1)], axis=2 )

        # for order in range(1,self.maxCompleteOrder+1):
        #     for q in range(order+1):
        #         p=order-q
        #         multcoefs.append( (p, q, dxy[p,q]) )
        return dL, dxy #, multcoefs

    def exportKicks(self, filename, crunchAfterSlice: int=-1, crunchBeforeSlice: int=0):
        crunchAfter: bool= crunchAfterSlice >= 0
        final = crunchAfterSlice if crunchAfter else len(self.s)-1
        nKicks = final - crunchBeforeSlice + 1

        with open(filename, 'w') as f:
            f.write('# kick file (PolyPotential)')
            #f.write(f'\nmaxCompleteOrder {self.maxCompleteOrder}')
            f.write(f'\norder {self.dxy.shape[0]} {self.dxy.shape[1]}')
            f.write(f'\ndriftlen {self.dL}')
            f.write(f'\nnkicks {nKicks}')

            def poXYkick(n: int, **kwargs):
                dL, dxy = self.getSlice(n, **kwargs)
                f.write(f'\nlen {dL} poXY')
                for coef in dxy.flat:
                    f.write(f' {coef}')
                #for coefs in multcoefs:
                #    f.write(f' {coefs[0]},{coefs[1]}:{coefs[2]}')

            if crunchBeforeSlice==0:
                poXYkick(0, half=self.startSymmetric)
            else:
                # note: half=False introduces minor L/2 error at symmetry-plane kick
                poXYkick(0, half=False, nEnd=crunchBeforeSlice)

            [poXYkick(n) for n in range(crunchBeforeSlice+1,final)]
            poXYkick(final, half=not crunchAfter, nEnd=-1 if crunchAfter else 0)

        startPos = self.dL*crunchBeforeSlice
        endPos = self.dL*final # (there are final+1 kicks with dL in between)

        print(f'{nKicks} kicks exported to {filename}, startPos (crunch before) {startPos} m, endPos (crunchAfter) {endPos} m, total length {endPos-startPos} m.')

    def integrated(self, p: int, q: int):
        return trapz(self.dxy[p,q], self.s)

    def plot(self, ax, order: int):
        for p in range(order+1):
            q = order-p
            try:
                ax.plot(self.s, self.dxy[p,q], label=f'psi({p},{q})')
            except IndexError:
                pass
        ax.set(xlabel='s [m]', ylabel=r'curvature [1/m]' if order==1 else f'focusing [1/m$^{order}$]', xlim=(self.s[0], self.s[-1]))
        ax.legend()

    def __repr__(self):
        stri = f'PolyPotential of order-shape {self.dxy.shape[0]} {self.dxy.shape[1]}\n'
        maxo = self.dxy.shape[0]+self.dxy.shape[1]-2
        for order in range(maxo):
            stri += f'  multipole order {order} (incomplete)\n' if order>=self.dxy.shape[0] else f'  multipole order {order}\n'
            for p in range(order+1):
                q = order-p
                try:
                    dxy = self.dxy[p,q]
                except IndexError:
                    continue
                i_dxy = self.integrated(p,q)
                stri += f'    ({p},{q}): maxabs={amax(absolute(dxy)):.4e}, avg={mean(dxy):.4e}, integrated={i_dxy}\n'
        return stri


class Trajectory():
    def __init__(self, Z, X, s, theta, curv=None, startSymmetric=True):
        self.s = copy(asarray(s).reshape([-1]))
        self.Z = copy(asarray(Z).reshape([-1]))
        self.X = copy(asarray(X).reshape([-1]))
        self.theta = copy(asarray(theta).reshape([-1]))
        self.genCTheta()
        self.curv = copy(asarray(curv).reshape([-1]))

        self.bFocus = None
        self.kFocus = None
        self.startSymmetric = startSymmetric

    def genCTheta(self):
        """assumes initial angle is zero (symmetric)"""
        self.cTheta = zeros_like(self.s)
        if len(self.s) > 1:
            self.cTheta[1:] = (self.theta[:-1] + self.theta[1:]) / 2
            self.cThetaSin = sin(self.cTheta)
            self.cThetaCos = cos(self.cTheta)

    def __len__(self):
        return self.s.size

    def track(self, sDiff: float, nSteps: int, curv_func, scale: float=1.0):
        nOld = len(self)
        nNew = nOld + nSteps
        self.s.resize(nNew)
        self.s[nOld:] = self.s[nOld-1] + arange(1, nSteps+1)*sDiff
        self.Z.resize(nNew)
        self.X.resize(nNew)
        self.curv.resize(nNew)
        self.theta.resize(nNew)
        for n in range(nOld, nNew):
            zNow = self.Z[n] = self.Z[n - 1] + sDiff * cos(self.theta[n - 1])
            xNow = self.X[n] = self.X[n - 1] - sDiff * sin(self.theta[n - 1])
            self.curv[n] = scale*curv_func(zNow, xNow)
            self.theta[n] = self.theta[n - 1] + sDiff * self.curv[n]
        self.genCTheta()

    def sqDistanceToPoint(self, pointZ, pointX, normalize=True, mm=True):
        "assumes ordered Z"
        posidx=argmax(self.Z>pointZ)
        #trajEdge = (self.Z[posidx-1]-self.Z[posidx], self.X[posidx-1]-self.X[posidx])
        #pointEdge = (pointZ-self.Z[posidx], pointX-self.X[posidx])
        # cross product
        edgeZ = (self.Z[posidx-1]-self.Z[posidx])*1000
        edgeX = (self.X[posidx-1]-self.X[posidx])*1000
        cross_prod = 1000*((pointZ-self.Z[posidx])*edgeX - (pointX-self.X[posidx])*edgeZ)
        cross_prod *= cross_prod
        if normalize:
            cross_prod /= edgeX**2 + edgeZ**2
        if not mm:
            cross_prod /= 1000
        return cross_prod

    def addFocusingTerms(self, curvatureFunction, rigidity=None, maxOrder=3):
        def b_transx(Z, X, nbeamxZ, nbeamxX, beamx):
            # coordinate transformation to form the directional derivative of B in direction of beamx,
            # nbeamx is a perpendicular unit vector to ntr, which points in s direction. Also, nbeamxX > 0. We have
            # We have nbeamxX = self.Zp, nbeamxZ = -self.Xp
            return curvatureFunction(Z + nbeamxZ * beamx, X + nbeamxX * beamx)

        bFocus = FocusingTerms(b_transx, self, name='b', rigidity=rigidity, maxOrder=maxOrder)

        def k_transx(Z, X, nbeamxZ, nbeamxX, beamx):
            # effective orbit force integrated with arc length
            return b_transx(Z, X, nbeamxZ, nbeamxX, beamx) * (1 + curvatureFunction(Z, X) * beamx)

        kFocus = FocusingTerms(k_transx, self, minOrder=1, name='k', rigidity=rigidity, maxOrder=maxOrder)
        kFocus.terms[0] = bFocus.terms[0]

        self.bFocus = bFocus
        self.kFocus = kFocus

    def plot(self, ax, whiteBg=True, linewidth=0.9, mirrorZ=False, **kwargs):
        zed = -self.Z if mirrorZ else self.Z
        if whiteBg:
            ax.plot(zed, self.X, linewidth=2.0, color='white')
        ax.plot(zed, self.X, linewidth=linewidth, **kwargs)
        ax.set(xlabel='Z [m]', ylabel='X [m]')

    def angle(self, n=-1, deg=True):
        ang = self.cTheta[n]
        return ang * 180 / pi if deg else ang

    def indexAtS(self, thisS):
        return argmax(self.s>thisS)

    def __repr__(self):
        return 'orbit trajectory starting (Z=%.2e m, X=%.2e m) ending (Z=%.2e m, X=%.2e m) with %.3f deg' \
               % (self.Z[0], self.X[0], self.Z[-1], self.X[-1], self.angle(deg=True))

    def toFrame(self):
        df = DataFrame()
        for key in ('s', 'Z', 'X', 'cTheta'):
            df[key] = getattr(self, key)
        df['cnZ'] = cos(self.cTheta)
        df['cnX'] = -sin(self.cTheta)
        if self.bFocus is not None:
            df = self.bFocus.toFrame(df)
        return df

    def toFile(self, filename: str):
        df = self.toFrame()
        df_print(df, filename)

def parallelTrajectory(s=arange(0,1,0.001), xOffset=0.0):
    z = arange(0,0.7,0.001)
    x = ones_like(z)*xOffset
    return Trajectory(z, x, z, zeros_like(z))


class CurvatureFunction2D():
    def __init__(self, beamEnergyMeV):
        self.subFields = list()
        self.rigidity = rigidityFromMeV(beamEnergyMeV)
        self.trajectories = list()

    def addField(self, field: AbstractField):
        field.normalize(self.rigidity)
        self.subFields.append(field)

    def curvatureInPlane(self, z, x):
        #b = zeros((z.size, x.size), dtype=float)
        zMesh, xMesh = [arr.T.flatten() for arr in meshgrid(z, x)]
        b = zeros_like(zMesh)
        for subField in self.subFields:
            b += subField.onPath(zMesh, xMesh)
        return b.reshape((z.size, x.size))

    def curvatureOnPath(self, z, x):
        b = zeros(z.size, dtype=float)
        for subField in self.subFields:
            b += subField.onPath(z, x)
        return b

    def potentialOnPath(self, x, y, z):
        pot = zeros(z.size, dtype=float)
        for subField in self.subFields:
            try:
                pot += subField.potentialOnPath(x, y, z)
            except NameError:
                raise RuntimeError('at least one subfield is not 3d')
        return pot

    def plot(self, ax, z=None, x=None, vlim=None, cmap='BrBG',
             trajectories=None):  # z=arange(-0.25, 1, 0.005), x=arange(-0.02, 0.02, 0.002)):
        if z is None:
            lim = zeros((2, 2), dtype=float)  # array([[0,1e-3],[-5e-3,5e-3]], float)  #xlim, ylim
            try:
                for trajectory in trajectories:
                    trajectory.plot(ax)
                    traj = array(((amin(trajectory.Z), amax(trajectory.Z)), (amin(trajectory.X), amax(trajectory.X))))
                    for n in range(2):
                        if traj[n, 0] < lim[n, 0]:
                            lim[n, 0] = traj[n, 0]
                        if traj[n, 1] > lim[n, 1]:
                            lim[n, 1] = traj[n, 1]
                lim += [[0, 1e-3], [-5e-3, 5e-3]]
                z = linspace(lim[0, 0], lim[0, 1], 250)
                x = linspace(lim[1, 0], lim[1, 1], 20)
            except TypeError:
                raise Exception('generate a trajectory for map limits.')

        fld = self.curvatureInPlane(z, x).T
        if vlim is None:
            vlim = amax(absolute(fld))
        levels = vlim * arange(-20.5, 21) / 20  #arange(-10.5, 11) / 10
        quad = ax.contourf(z, x, fld, levels, cmap=cmap)  # vmin=-vlim, vmax=vlim
        # cBar = colorbar(quad)
        # cBar.set_label('curvature $\cdot$ m')

    def solveTrajectory(self, X0: float, sMax: float, sDiff: float=1e-3, scale: float=1.0):
        """
        generate a Trajectory starting at (X=X0,Z=0) with edge length sDiff and up to length sMax
        """
        curv0 = scale*self.curvatureOnPath(asarray(0.0), asarray(X0))
        traj = Trajectory(0.0, X0, 0.0, curv0*sDiff/2, curv0)
        s = arange(0, sMax + sDiff / 2, sDiff)
        traj.track(sDiff, len(s)-1, self.curvatureOnPath, scale)
        return traj

    def findXforAngle(self, designAngleDeg: float, sMax: float, sDiff: float=1e-3, xSearchInterval=(-0.01, 0.01), **kwargs):
        """
        generate a Trajectory ending with designAngleDeg with edge length dFiff and up to length sMax. (optimizes using .solveTrajectory)
        """
        def angleDiff(X: float) -> float:
            traj = self.solveTrajectory(X, sMax, sDiff)
            return traj.angle(deg=True)-designAngleDeg
        result = root_scalar(angleDiff, bracket=xSearchInterval, **kwargs)
        print(result)
        return self.solveTrajectory(result.root, sMax, sDiff)

    def findXthruPoint(self, pointZX: iter, sMax: float, sDiff: float=1e-3, **kwargs):
        """
        generate a Trajectory running through pointZX with edge length dFiff and up to length sMax. (optimizes using .solveTrajectory)
        """
        def squareDist(X: float) -> float:
            traj = self.solveTrajectory(X, sMax, sDiff)
            idex = argmin(absolute(traj.Z-pointZX[0]))
            return (traj.Z[idex]-pointZX[0])**2 + (traj.X[idex]-pointZX[1])**2
        result = minimize_scalar(squareDist, **kwargs)
        print(result)
        return self.solveTrajectory(result.x, sMax, sDiff)

    def addFocusingToTrajectory(self, trajectory: Trajectory, maxOrder=3):
        trajectory.addFocusingTerms(self.curvatureOnPath, rigidity=self.rigidity, maxOrder=maxOrder)


def dummyKickDrift(filename, length = 0.2025, iRho = 1.49969e-01, N=32):
    sliceLength = length/N
    with open(filename, 'w') as f:
        f.write("# kick-drift export (upright multipoles) for tracy-null\norder 2\n")
        f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000\ndrif %.8e\n' % (sliceLength/2, iRho, sliceLength))
        for n in range(1,N):
            f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000\ndrif %.8e\n' % (sliceLength, iRho, sliceLength))
        f.write('kick len %.8e hRef %.8e mpa 0.00000000e+00 0.0000' % (sliceLength/2, iRho))
