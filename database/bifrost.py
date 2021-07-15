from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

def iterate_ctx_crd(ctx, Nscatter=10, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices(verbose=True)
        if i < Nscatter:
            continue
        delta = ctx.stat_equil(printUpdate=True)

        if dJ < 3e-3 and delta < 1e-3:
            print(i)
            print('----------')
            return

def synth_spectrum(atmos, depthData=False, Nthreads=1, conserveCharge=False, allactive=True):
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(),
    C_atom(),
     OI_ord_atom(), Si_atom(), Al_atom(),
    CaII_atom(),
    Fe_atom(),
    He_9_atom(),
    MgII_atom(), N_atom(), Na_atom(), S_atom()
    ])
    if (allactive):
        aSet.set_active('H', 'Ca')
    else:
        aSet.set_active('Ca')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)

    ctx = lw.Context(atmos, spect, eqPops, ngOptions=lw.utils.NgOptions(0,0,0), Nthreads=Nthreads, conserveCharge=conserveCharge)
    if depthData:
        ctx.depthData.fill = True
    iterate_ctx_crd(ctx)
    eqPops.update_lte_atoms_Hmin_pops(atmos)
    ctx.formal_sol_gamma_matrices()
    return ctx

atmosRef = Falc82()
# ctxRef = synth_spectrum(atmosRef, depthData=True, conserveCharge=True)

fmodel = fits.open('/net/drogon/scratch1/aasensio/3dcubes/Enhanced_network_385_tau_from_RH_01_tau8.fits')
x, y = 50, 20
bifrost = fmodel[0].data[:, :, x, y].astype('<f8')

# tau, T, Pe, vmicro, B, vlos, theta, azimuth, z, Pgas, rho_gas


tau500 = np.ascontiguousarray(10.0**bifrost[0, ::-1])
T = np.ascontiguousarray(bifrost[1, ::-1])
vlos = np.ascontiguousarray(bifrost[5, ::-1]) / 100.0  # m/s
vturb = np.ascontiguousarray(bifrost[3, ::-1]) / 100.0 # m/s
Pe = np.ascontiguousarray(bifrost[2, ::-1])
Ne = Pe / (1.381e-16 * T) * 1e6    # m-3

atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.Tau500, depthScale=tau500, temperature=T, vlos=vlos, vturb=vturb, ne = Ne, verbose=True)
ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False, allactive=True)
ctx2 = synth_spectrum(atmos, depthData=True, conserveCharge=False, allactive=False)

# cmass_max = 1.8
# cmass_min = -4.0
# n = 82
# cmass = np.linspace(cmass_min, cmass_max, n)

# f = int.interp1d(np.log10(atmosRef.cmass), atmosRef.temperature, bounds_error=False, fill_value=(atmosRef.temperature[0], atmosRef.temperature[-1]))
# Tnew = f(cmass)
# f = int.interp1d(np.log10(atmosRef.cmass), atmosRef.vlos, bounds_error=False, fill_value=(atmosRef.vlos[0], atmosRef.vlos[-1]))
# vlosnew = f(cmass)
# f = int.interp1d(np.log10(atmosRef.cmass), atmosRef.vturb, bounds_error=False, fill_value=(atmosRef.vturb[0], atmosRef.vturb[-1]))
# vturbnew = f(cmass)

# atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=10**cmass, temperature=Tnew, vlos=vlosnew, vturb=vturbnew, verbose=True)
# ctx = synth_spectrum(atmos, depthData=True, conserveCharge=True)

plt.ion()
# plt.plot(ctx.spect.wavelength, (ctxFast.spect.I[:, -1] - ctx.spect.I[:, -1]) / ctxFast.spect.I[:, -1])
plt.plot(ctx.spect.wavelength, ctx.spect.I[:, -1])
plt.plot(ctx2.spect.wavelength, ctx2.spect.I[:, -1])
# plt.plot(ctx.spect.wavelength, ctxPyTau.spect.I[:, -1])
# plt.plot(ctx.spect.wavelength, ctxRef.spect.I[:, -1], '--')
plt.xlim(853.9444, 854.9444)
