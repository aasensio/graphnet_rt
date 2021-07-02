from lightweaver.fal import Falc82
from lightweaver.rh_atoms import H_6_atom, H_6_CRD_atom, H_3_atom, C_atom, O_atom, OI_ord_atom, Si_atom, Al_atom, CaII_atom, Fe_atom, FeI_atom, He_9_atom, He_atom, He_large_atom, MgII_atom, N_atom, Na_atom, S_atom
import lightweaver as lw
import numpy as np
import scipy.interpolate as int
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
            # print(i)
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
    aSet.set_active('H', 'Ca')
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
    def __init__(self):
        self.atmosRef = Falc82()
        self.ltau = np.log10(self.atmosRef.tauRef)

        self.ltau_nodes = np.array([np.min(self.ltau), -5, -4, -3, -2, -1, 0, np.max(self.ltau)])
        self.ntau = len(self.ltau_nodes)

        self.ind_ltau = np.searchsorted(self.ltau, self.ltau_nodes)

    def new_model(self):
        scale = np.linspace(2500.0, 2500.0, self.ntau)
        deltas = np.random.normal(loc=0.0, scale=scale, size=self.ntau)

        deltas_smooth = smooth(deltas, 2)
        f = int.interp1d(self.ltau_nodes, deltas_smooth, kind='quadratic', bounds_error=False, fill_value="extrapolate")
        T_new = self.atmosRef.temperature + f(self.ltau)
        # T_new[T_new < 3000] = 3000.0
        
        atmos = lw.Atmosphere.make_1d(scale=lw.ScaleType.ColumnMass, depthScale=self.atmosRef.cmass, temperature=T_new, vlos=self.atmosRef.vlos, vturb=self.atmosRef.vturb, verbose=False)

        return atmos

    def solve_NLTE(self, atmos):
        ctx = synth_spectrum(atmos, depthData=True, conserveCharge=True)

    def gen_database(self, n):

        new_atmos = self.new_model()

        self.ctx = synth_spectrum(new_atmos, depthData=True, conserveCharge=True)

        log_departure = [None] * 2
        for i in range(2):
            log_departure[i] = np.log10(tmp.ctx.activeAtoms[i].n / tmp.ctx.activeAtoms[i].nStar)

        return log_departure
    
def master_work(n, filename, write_frequency=1):
    db = Database()

    task_index = 0
    num_workers = size - 1
    closed_workers = 0

    log_departure_list = [None] * n
    T_list = [None] * n
    tau_list = [None] * n
    cmass = [None]

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

                    pbar.set_postfix(sent=f'{task_index}->{source}')
                
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

    with open(f'{filename}_tau.pk', 'wb') as filehandle:
        pickle.dump(tau_list, filehandle)
    

def slave_work():
    db = Database()

    while True:
        comm.send(None, dest=0, tag=tags.READY)
        dataReceived = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)

        tag = status.Get_tag()
            
        if tag == tags.START:            
            # Do the work here
            task_index = dataReceived['index']
                    
            success = 1

            log_departure = [None] * 2
            tau = None
            cmass = None
            temperature = None

            try:
                atmos = db.new_model()
                ctx = synth_spectrum(atmos, depthData=True, conserveCharge=True)
                tau = atmos.tauRef
                cmass = atmos.cmass
                temperature = atmos.temperature
                for i in range(2):
                    log_departure[i] = np.log10(ctx.activeAtoms[i].n / ctx.activeAtoms[i].nStar)
            except:
                success = 0           
            
                
            dataToSend = {'index': task_index, 'T': temperature, 'log_departure': log_departure, 'tau': tau, 'cmass': cmass, 'success': success}

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
           
    if rank == 0:        
        parser = argparse.ArgumentParser(description='Generate synthetic models and solve NLTE problem')
        parser.add_argument('--n', '--n', default=10000, type=int, metavar='N', help='Number of models')
        parser.add_argument('--f', '--freq', default=1, type=int, metavar='FREQ', help='Frequency of model write')
        parser.add_argument('--o', '--out', default='training', type=str, metavar='OUTFILE', help='Root of output files')

        parsed = vars(parser.parse_args())

        master_work(parsed['n'], parsed['out'], write_frequency=parsed['freq'])
    else:
        slave_work()
