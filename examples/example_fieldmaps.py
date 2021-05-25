from numpy import arange, copy, empty, savez, meshgrid, pi, sin, cos, sinh, cosh, sqrt, arctan2, zeros_like
from trap.ZX.basic import rigidityFromMeV
from scipy.special import iv, ivp
# units in mm
xRng = arange(-20,21)
yRng = arange(11) # the first index must hold y=0
zRng = arange(300)

def saveMagneticFieldmap(filename: str, bFull):
    savez(filename, bFull=copy(bFull), xRng=xRng, yRng=yRng, zRng=zRng) # copy is just a precaution, should work without
    print(f'fieldmap saved to {filename}.')


bFull = empty((len(xRng), len(yRng), len(zRng), 3))

x, y, z = meshgrid(xRng/1000, yRng/1000, zRng/1000, indexing='ij')  # used to construct fields, unit is [m]

## case 1: dipole, bY=2 T everywhere
bFull[:] = 0
bFull[:,:,:,1] = 2.0  # bY
saveMagneticFieldmap('pure_dipole.npz', bFull)


## case 2: quadrupole
k = 5.0 
quadgrad = k*rigidityFromMeV(2700)
bFull[:] = 0
bFull[:,:,:,1] = quadgrad*x
bFull[:,:,:,0] = quadgrad*y  # bX
saveMagneticFieldmap('pure_quad.npz', bFull)

## case 3: square-approximation dipole
# potential: coef cos(kappa_n z) sinh(kappa_n y) / kappa_n
coefs = (8/9, 1, 0, -1/9)
L = 0.25 # 20cm
bFull[:] = 0
for n, coef in enumerate(coefs):
    kappan = n*pi / L
    if n==0:
    	bFull[:,:,:,1] = coef
    else:
        bFull[:,:,:,1] += coef *cos(kappan*z) *cosh(kappan*y)
        bFull[:,:,:,2] -= coef *sin(kappan*z) *sinh(kappan*y)  # bZ
saveMagneticFieldmap('sqa_dipole.npz', bFull)

## case 4: square-approximation quadrupole
# potential: 4 coef cos(kappa_n z) iv(n, kappa_n r) sin(m ph) / kappa_n^2
bFull[:] = 0
r = sqrt(x**2+y**2)
ph = arctan2(y,x)
m=2
bFull[:] = 0
for n, coef in enumerate(coefs):
    kappan = n*pi / L
    if n==0:
        bFull[:,:,:,1] = coef*x
        bFull[:,:,:,0] = coef*y
    else:
        bR  = 4*coef *cos(kappan*z) *ivp(m,kappan*r) *sin(m*ph) / kappan
        ivmr = zeros_like(bR)
        ivmr[r!=0] = iv(m,kappan*r[r!=0])  / r[r!=0]
        bPh = 4*m*coef *cos(kappan*z) *ivmr * cos(m*ph) / kappan**2

        bFull[:,:,:,0] +=  bR*cos(ph) - bPh*sin(ph)  # bX
        bFull[:,:,:,1] +=  bR*sin(ph) + bPh*cos(ph)  # bY
        bFull[:,:,:,2] -= 4*coef *sin(kappan*z) *iv(m,kappan*r) *sin(m*ph) / kappan
saveMagneticFieldmap('sqa_quad.npz', bFull)
 
