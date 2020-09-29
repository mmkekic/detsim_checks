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

filename = 'NEXT100.tracking.S2.SiPM.LightTable.h5'
drift_velocity_EL = 2.5 * units.mm/units.mus
wf_sipm_bin_width = 1 * units.mus
wf_buffer_length  = 2000  * units.mus


xsipms, ysipms = datasipm["X"].values, datasipm["Y"].values                                     
nsipms = len(datasipm)


from detsim_loop import electron_loop 
from psf_functions import PSF_distance as PSF_distance_

from invisible_cities.detsim.psf_functions import PSF_distance as PSF_distance

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
def cython_class(times, nphotons, dx, dy):
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


n_els = 100000

times = np.random.sample(n_els)*0.9*wf_buffer_length
nphotons = np.random.choice(np.arange(20000), n_els)
dx = xsipms.min()+np.random.sample(n_els)*(xsipms.max()-xsipms.min())
dy = ysipms.min()+np.random.sample(n_els)*(xsipms.max()-ysipms.min())

t0 = time.time()
sipm_org = np.asarray(cython_setup_org(times, nphotons, dx, dy))
torg = time.time()
sipm_mod = np.asarray(cython_setup_complic(times, nphotons, dx, dy))
tmod = time.time()
sipm_fin = np.asarray(final_cython_class(times, nphotons, dx, dy))
tfin = time.time()
print(np.all(sipm_org==sipm_mod))
print(np.all(sipm_org==sipm_fin))
print (np.sum(sipm_org), np.sum(sipm_mod), np.sum(sipm_fin))
print(torg-t0, tmod-torg, tfin-tmod)
# # import timeit, functools
# # t = timeit.Timer(functools.partial(cython_setup, times, nphotons, dx, dy)) 
# print (t.timeit(5))
