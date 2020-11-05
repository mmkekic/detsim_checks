import numpy as np
import pandas as pd
import line_profiler
import atexit
profile = line_profiler.LineProfiler()
atexit.register(profile.print_stats)
import time

from invisible_cities.core import system_of_units as units                                          
from IPython.core.debugger import set_trace
import invisible_cities.database.load_db as db
datasipm = db.DataSiPM('next100', 0)

filename = './Lighttables/NEXT100.tracking.S2.SiPM.LightTable.h5'
drift_velocity_EL = 2.5 * units.mm/units.mus
wf_sipm_bin_width = 1 * units.mus
wf_pmt_bin_width = 25 * units.ns
wf_buffer_length  = 2000  * units.mus


xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values                                     
nsipms = len(datasipm)


from detsim_loop import electron_loop 
from psf_functions import PSF_distance as PSF_distance_



from detsim_loop_original import electron_loop as el_loop_org
from detsim_loop_modular import electron_loop as el_loop_mod
def create_sipm_waveforms_cython(wf_buffer_length  : float,
                                 wf_sipm_bin_width : float,
                                 ELtimes   : np.ndarray,
                                 datasipm  : pd.DataFrame,
                                 PSF):
    nsipms = len(datasipm)
    xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values
    sipm_time_bins = np.arange(0, wf_buffer_length, wf_sipm_bin_width)
    PSF_distances = PSF.index.values
    PSF_values = PSF.values
    #@profile
    def create_sipm_waveforms_(times,
                               photons,
                               dx,
                               dy):
        
        sipmwfs =  el_loop_org(dx.astype(np.float64), dy.astype(np.float64), times.astype(np.float64), photons.astype(np.uint),
                                 xsipms.astype(np.float64), ysipms.astype(np.float64), PSF_values.astype(np.float64), PSF_distances.astype(np.float64), ELtimes.astype(np.float64), wf_sipm_bin_width, wf_buffer_length)

        return sipmwfs
    return create_sipm_waveforms_

@profile
def cython_setup_class(times, nphotons, dx, dy):
    psf_cl = PSF_distance_(datasipm, filename)
    drift_velocity_EL = 2.5 * units.mm/units.mus
    nlen = wf_buffer_length//wf_sipm_bin_width
    vals = electron_loop(dx, dy, times, nphotons.astype(np.uint),  psf_cl, drift_velocity_EL, wf_sipm_bin_width, nlen)
    return vals



def cython_setup_modular(times, nphotons, dx, dy):
    drift_velocity_EL = 2.5 * units.mm/units.mus
    nlen = wf_buffer_length//wf_sipm_bin_width
    vals = el_loop_mod(dx, dy, times, nphotons.astype(np.uint),  datasipm, filename, drift_velocity_EL, wf_sipm_bin_width, nlen)
    return vals



def cython_setup_org(times, nphotons, dx, dy):
    PSF    = pd.read_hdf(filename, "/LightTable")
    Config = pd.read_hdf(filename, "/Config")
    EL_dz = float(Config.loc["EL_GAP"])           * units.mm
    el_pitch = float(Config.loc["pitch_z"].value) * units.mm
    ELtimes = np.arange(el_pitch/2., EL_dz, el_pitch)/drift_velocity_EL
    create_sipm_waveform = create_sipm_waveforms_cython(wf_buffer_length//wf_sipm_bin_width, wf_sipm_bin_width, ELtimes, datasipm, PSF)
    result = create_sipm_waveform(times, nphotons, dx, dy)
    return result

from IPython.core.debugger import set_trace

def multinomial_approach(times, nphotons, dx, dy):
    datasipm = db.DataSiPM('next100', 0)
    filename = './Lighttables/NEXT100.tracking.S2.SiPM.LightTable.h5'
    drift_velocity_EL = 2.5 * units.mm/units.mus
    wf_sipm_bin_width = 1 * units.mus
    wf_buffer_length  = 2000  * units.mus
    xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values                                     
    nsipms = len(datasipm)
    
    sipmwfs = np.zeros([nsipms, int(wf_buffer_length//wf_sipm_bin_width)])
    sipm_timebin = np.arange(0, wf_buffer_length, wf_sipm_bin_width)
    
    PSF    = pd.read_hdf(filename, "/LightTable")
    Config = pd.read_hdf(filename, "/Config")
    EL_dz = float(Config.loc["EL_GAP"])           * units.mm
    el_pitch = float(Config.loc["pitch_z"].value) * units.mm
    EL_times = np.arange(el_pitch/2., EL_dz, el_pitch)/drift_velocity_EL
    npartitions = PSF.shape[1]
    
    zbins = np.arange(0, npartitions)
    
    numel = len(dx)
    max_psf = 200
    probs = np.zeros(shape=(npartitions, nsipms+1)) #last sipm is no-detection    
    for elindx in range(numel):

        xel = dx[elindx]
        yel = dy[elindx]
        tel = times[elindx]
        phel = nphotons[elindx]
        tint = times[elindx]
        dist = np.sqrt((xel-xsipms)**2 + (yel-ysipms)**2)
        dist_msk = dist<max_psf
        
        
        psf_indcs = np.digitize(dist[dist_msk], PSF.index)-1
        dist_msk = np.append(dist_msk, [False]) #to match nsipm+1 shape
        probs[:, dist_msk] = PSF.values[psf_indcs].T/npartitions 
        
        signal = np.random.multinomial(phel, probs.ravel(), size=1).reshape(npartitions, nsipms+1)[:, :-1]#remove no-detected column
        

        time_indx = np.digitize(tint + EL_times, sipm_timebin)-1
        np.add.at(sipmwfs, (slice(0, nsipms), time_indx), signal.T)
        
    return sipmwfs

np.random.seed(0)
n_els = 20000

times = np.random.normal(100*units.mus, 4, n_els)
nphotons = np.random.normal(500, 10, n_els).astype(int)
dx = np.random.normal(0, 40, n_els)
dy = np.random.normal(0, 40, n_els)

# sipms_pre_poiss = np.asarray(cython_setup_org(times, nphotons, dx, dy))

# sipm_org = np.random.poisson(np.asarray(cython_setup_org(times, nphotons, dx, dy)))
#sipm_just = multinomial_approach(times, nphotons, dx, dy)

# t0 = time.time()
# sipm_org = np.asarray(cython_setup_org(times, nphotons, dx, dy))
# torg = time.time()
# sipm_fin = np.asarray(cython_setup_class(times, nphotons, dx, dy))
# tfin = time.time()
# sipm_mod = 0#np.asarray(cython_setup_modular(times, nphotons, dx, dy))
# tmod = time.time()
# print(np.all(sipm_org==sipm_mod))
# print(np.all(sipm_org==sipm_fin))
# # #print(np.all(sipm_org==sipm_fin))
# # #print (np.sum(sipm_org), np.sum(sipm_mod), np.sum(sipm_fin))
# print(torg-t0, tmod-tfin, tfin-torg)
# # # # import timeit, functools
# # # # t = timeit.Timer(functools.partial(cython_setup, times, nphotons, dx, dy)) 
# # # print (t.timeit(5))



# PSF    = pd.read_hdf(filename, "/LightTable")

# prob_det = 0.0004

# N_phot = 500*2000

# num_im_sipms = 9
# num_tbins = 4

# N_ph_per_sens_per_tb = int(N_phot/(num_im_sipms*num_tbins))

# ml_dist = np.random.multinomial(N_ph_per_sens_per_tb, [prob_det/(num_im_sipms*num_tbins), 1-prob_det], 10000)[:, 0]


# poiss_dist = np.random.poisson(N_ph_per_sens_per_tb*prob_det/(num_im_sipms*num_tbins), 10000)

# from matplotlib import pyplot as plt

# plt.hist(ml_dist, label='ml', bins = np.linspace(0.9*poiss_dist.min(), 1.1*poiss_dist.max(), 80), histtype='step')
# plt.hist(poiss_dist, label='poiss', bins = np.linspace(0.9*poiss_dist.min(), 1.1*poiss_dist.max(), 80), histtype='step')
# plt.legend()
# plt.show()

from psf_functions import LTS2
filename_s2lt = '/home/mmkekic/NEXT/NEXT_data/detsim_debug/Lighttables/NEXT100.energy.S2.PmtR11410.LightTable.h5'

from invisible_cities.cities.detsim_source             import load_MC
from invisible_cities.cities.detsim_simulate_electrons import generate_ionization_electrons as generate_ionization_electrons_
from invisible_cities.cities.detsim_simulate_electrons import drift_electrons               as drift_electrons_
from invisible_cities.cities.detsim_simulate_electrons import diffuse_electrons             as diffuse_electrons_

from invisible_cities.cities.detsim_simulate_signal    import pes_at_pmts
from invisible_cities.cities.detsim_simulate_signal    import generate_S1_times_from_pes    as generate_S1_times_from_pes_

from invisible_cities.cities.detsim_waveforms          import create_pmt_waveforms          as create_pmt_waveforms_
from invisible_cities.cities.detsim_waveforms          import create_sipm_waveforms         as create_sipm_waveforms_

from invisible_cities.cities.detsim_get_psf            import create_lighttable_function
from invisible_cities. detsim import psf_functions  as psf
el_gain                = 500

ws = 39.2 * units.eV
wi = 22.4 * units.eV
fano_factor = 0.15
conde_policarpo_factor = 1.00




wf_buffer_length       = 2000 * units.mus
wf_pmt_bin_width       =   25 * units.ns
wf_sipm_bin_width      =    1 * units.mus



S2_LT = create_lighttable_function(filename_s2lt)#s2_lighttable)

Config = pd.read_hdf(filename, "/Config")
EL_dz    = float(Config.loc["EL_GAP"])        * units.mm
el_pitch = float(Config.loc["pitch_z"].value) * units.mm
el_gain_sigma = np.sqrt(el_gain * conde_policarpo_factor)

EL_dtime      =  EL_dz / drift_velocity_EL
s2_pmt_nsamples  = max((int(EL_dtime // wf_pmt_bin_width ), 1))


def gonzalo_setup(times, nphotons, dx, dy):
    S2pes_at_pmts = pes_at_pmts(S2_LT, nphotons, dx, dy)
    create_S2pmtwfs = create_pmt_waveforms_("S2", wf_buffer_length, wf_pmt_bin_width)
    waveform = create_S2pmtwfs(times, S2pes_at_pmts)
    return waveform
@profile
def cython_setup_class_lt(times, nphotons, dx, dy):
    drift_velocity_EL = 2.5 * units.mm/units.mus
    EL_gap = 10*units.mm
    zbins = np.arange(0, EL_dtime +wf_pmt_bin_width, wf_pmt_bin_width)
    psf_cl = LTS2(filename_s2lt, zbins)
    print('created class')
    nlen = wf_buffer_length//wf_pmt_bin_width
    vals = electron_loop(dx, dy, times, nphotons.astype(np.uint),  psf_cl, drift_velocity_EL, wf_pmt_bin_width, nlen)
    return vals

t0 = time.time()
vals = np.asarray(cython_setup_class_lt(times, nphotons, dx, dy))
tfim = time.time()
print(tfim-t0)

gon = gonzalo_setup(times, nphotons, dx, dy)
print(time.time()-tfim)
