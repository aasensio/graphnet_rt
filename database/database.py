from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import numpy as np
import scipy.interpolate as interp
from astropy.convolution import Box1DKernel
from astropy.convolution import convolve
from enum import IntEnum
from mpi4py import MPI
from tqdm import tqdm
import pickle
import argparse

class tags(IntEnum):
    READY = 0
    DONE = 1
    EXIT = 2
    START = 3

def smooth(sig, width):
    return convolve(sig, Box1DKernel(width))

def iterate_ctx_crd(ctx, Nscatter=10, NmaxIter=500):
    for i in range(NmaxIter):
        dJ = ctx.formal_sol_gamma_matrices(verbose=False)
        if i < Nscatter:
            continue
        delta = ctx.stat_equil(printUpdate=False)

        if dJ < 3e-3 and delta < 1e-3:
            # print(i, flush=True)
            # print('----------')
            return

def synth_spectrum(atmos, depthData=False, Nthreads=1, conserveCharge=False):
    atmos.quadrature(5)
    aSet = lw.RadiativeSet([H_6_atom(),
    C_atom(),
     OI_ord_atom(), Si_atom(), Al_atom(),
    CaII_atom(),
    Fe_atom(),
    He_9_atom(),
    MgII_atom(), N_atom(), Na_atom(), S_atom()
    ])
    # aSet.set_active('H', 'Ca')
    aSet.set_active('Ca')
    spect = aSet.compute_wavelength_grid()

    eqPops = aSet.compute_eq_pops(atmos)

    ctx = lw.Context(atmos, spect, eqPops, Nthreads=Nthreads, conserveCharge=conserveCharge)
    if depthData:
        ctx.depthData.fill = True
    
    iterate_ctx_crd(ctx)
    eqPops.update_lte_atoms_Hmin_pops(atmos, quiet=True)
    ctx.formal_sol_gamma_matrices(verbose=False)
    return ctx

class Database(object):
    def __init__(self, rank):
        # Read all available atmospheric models
        if (rank == 0):
            print(f"Reading FALC... ", flush=True)
        _, atmos_FALC = lw.multi.read_multi_atmos('semiempirical/FALC_82.atmos')
        if (rank == 0):
            print(f"Reading FALXCO...", flush=True)
        _, atmos_FALXCO = lw.multi.read_multi_atmos('semiempirical/FALXCO_80.atmos')
        if (rank == 0):
            print(f"Reading FALA...", flush=True)
        _, atmos_FALA = lw.multi.read_multi_atmos('semiempirical/FALA_80.atmos')
        if (rank == 0):
            print(f"Reading FALF... ", flush=True)
        _, atmos_FALF = lw.multi.read_multi_atmos('semiempirical/FALF_80.atmos')
        
        self.atmosRef = [atmos_FALC, atmos_FALXCO, atmos_FALA, atmos_FALF]

        self.n_atmos = len(self.atmosRef)

        self.ltau = [None] * self.n_atmos
        self.ltau_nodes = [None] * self.n_atmos
        self.ntau = [None] * self.n_atmos
        self.ind_ltau = [None] * self.n_atmos
        self.logT = [None] * self.n_atmos

        for i in range(self.n_atmos):
            self.ltau[i] = np.log10(self.atmosRef[i].tauRef)

            self.ltau_nodes[i] = np.array([np.min(self.ltau[i]), -5, -4, -3, -2, -1, 0, np.max(self.ltau[i])])
            self.ntau[i] = len(self.ltau_nodes[i])

            self.ind_ltau[i] = np.searchsorted(self.ltau[i], self.ltau_nodes[i])
            self.logT[i] = np.log10(self.atmosRef[i].temperature)

    def new_model(self):

        i = np.random.randint(low=0, high=self.n_atmos, size=1)[0]

        scale = np.linspace(2500.0, 2500.0, self.ntau[i])
        deltas = np.random.normal(loc=0.0, scale=scale, size=self.ntau[i])
        
        deltas_smooth = smooth(deltas, 2)
        f = interp.interp1d(self.ltau_nodes[i], deltas_smooth, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        T_new = self.atmosRef[i].temperature + f(self.ltau[i])
        # T_new[T_new < 3000] = 3000.0

        # Perturb vturb by 20% of the current value
        deltas_vturb = np.random.normal(loc=0.0, scale=0.2*self.atmosRef[i].vturb[self.ind_ltau[i]], size=self.ntau[i])
        f = interp.interp1d(self.ltau_nodes[i], deltas_vturb, kind='quadratic', bounds_error=False, fill_value="extrapolate")        
        vturb_new = self.atmosRef[i].vturb + f(self.ltau[i])
        
        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=self.atmosRef[i].cmass, temperature=T_new, vlos=self.atmosRef[i].vlos, vturb=vturb_new, verbose=False)

        return atmos
    
def master_work(n, filename, write_frequency=1):
    db = Database(rank=0)

    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    log_departure_list = [None] * n
    T_list = [None] * n
    tau_list = [None] * n
    vturb_list = [None] * n
    cmass = [None]
    success = True

    tasks = [0] * n
    pointer = 0
    n_done = 0
    
    print("*** Master starting with {0} workers".format(num_workers))
    with tqdm(initial=pointer, total=n, ncols=140) as pbar:
        while closed_workers < num_workers:
            dataReceived = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)                
            source = status.Get_source()
            tag = status.Get_tag()

            if tag == tags.READY:
                # Worker is ready, so send it a task
                try:                    
                    task_index = tasks.index(0)                    

                    dataToSend = {'index': task_index}
                    comm.send(dataToSend, dest=source, tag=tags.START)

                    tasks[task_index] = 1

                    if (success):
                        pbar.set_postfix(sent=f'{task_index}->{source}')
                    else:
                        pbar.set_postfix(sent=f'{task_index}R->{source}')
                
                except:
                    comm.send(None, dest=source, tag=tags.EXIT)

            elif tag == tags.DONE:
                index = dataReceived['index']                
                success = dataReceived['success']
                
                if (not success):
                    tasks[index] = 0
                else:
                    log_departure_list[index] = dataReceived['log_departure']
                    T_list[index] = dataReceived['T']
                    tau_list[index] = dataReceived['tau']
                    vturb_list[index] = dataReceived['vturb']
                    cmass = dataReceived['cmass']
                    pbar.update(1)                
                    
            elif tag == tags.EXIT:
                print(" * MASTER : worker {0} exited.".format(source))
                closed_workers += 1

            if (pbar.n / write_frequency == pbar.n // write_frequency):

                with open(f'{filename}_logdeparture.pk', 'wb') as filehandle:
                    pickle.dump(log_departure_list[0:task_index], filehandle)

                with open(f'{filename}_T.pk', 'wb') as filehandle:
                    pickle.dump(T_list[0:task_index], filehandle)
                
                with open(f'{filename}_vturb.pk', 'wb') as filehandle:
                    pickle.dump(vturb_list[0:task_index], filehandle)

                with open(f'{filename}_tau.pk', 'wb') as filehandle:
                    pickle.dump(tau_list[0:task_index], filehandle)

                with open(f'{filename}_cmass.pk', 'wb') as filehandle:
                    pickle.dump(cmass, filehandle)

    print("Master finishing")

    with open(f'{filename}_cmass.pk', 'wb') as filehandle:
        pickle.dump(cmass, filehandle)

    with open(f'{filename}_logdeparture.pk', 'wb') as filehandle:
        pickle.dump(log_departure_list, filehandle)

    with open(f'{filename}_T.pk', 'wb') as filehandle:
        pickle.dump(T_list, filehandle)

    with open(f'{filename}_vturb.pk', 'wb') as filehandle:
        pickle.dump(vturb_list, filehandle)

    with open(f'{filename}_tau.pk', 'wb') as filehandle:
        pickle.dump(tau_list, filehandle)
    

def slave_work(rank):
    db = Database(rank)

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)        

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
                    
            success = 1

            log_departure = None
            tau = None
            cmass = None
            temperature = None
            vturb = None

            try:
                atmos = db.new_model()
                ctx = synth_spectrum(atmos, depthData=True, conserveCharge=False)
                tau = atmos.tauRef
                cmass = atmos.cmass
                temperature = atmos.temperature
                vturb = atmos.vturb
                log_departure = np.log10(ctx.activeAtoms[0].n / ctx.activeAtoms[0].nStar)
            except:
                success = 0           
                            
            dataToSend = {'index': task_index, 'T': temperature, 'log_departure': log_departure, 'tau': tau, 'cmass': cmass, 'vturb': vturb, 'success': success}

            comm.send(dataToSend, dest=0, tag=tags.DONE)

        elif tag == tags.EXIT:
            break

    comm.send(None, dest=0, tag=tags.EXIT)
    

if (__name__ == '__main__'):

    # Initializations and preliminaries
    comm = MPI.COMM_WORLD   # get MPI communicator object
    size = comm.size        # total number of processes
    rank = comm.rank        # rank of this process
    status = MPI.Status()   # get MPI status object

    print(f"Node {rank}/{size} active", flush=True)
           
    if rank == 0:        
        parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
        parser.add_argument('--n', '--nmodels', default=10000, type=int, metavar='NMODELS', help='Number of models')
        parser.add_argument('--f', '--freq', default=1, type=int, metavar='FREQ', help='Frequency of model write')
        parser.add_argument('--o', '--outfile', default='training', metavar='OUTFILE', help='Root of output files')

        parsed = vars(parser.parse_args())
        
        master_work(parsed['n'], parsed['o'], write_frequency=parsed['f'])
    else:
        slave_work(rank)
