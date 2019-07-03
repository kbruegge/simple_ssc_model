from naima.models import InverseCompton, Synchrotron, LogParabola, TableModel
import numpy as np
import astropy.units as u
import click
from itertools import product
from joblib import Parallel, delayed
from astropy import constants
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator


class Log10Parabola:
    """Log-parabolic energy spectrum."""

    def __init__(self, amplitude=1E-12 * u.Unit('cm-2 s-1 TeV-1'), alpha=2, beta=1, reference=10 * u.TeV):
        self.amplitude = amplitude
        self.reference = reference
        self.alpha = alpha
        self.beta = beta
    
    def __call__(self, energy):
        return self.evaluate(energy, self.amplitude, self.alpha, self.beta, self.reference)

    def evaluate(energy, amplitude, alpha, beta, reference):
        xx = energy / reference
        exponent = -alpha - beta * np.log10(xx)
        return amplitude * np.power(xx, exponent)



def heaviside(energy, flux, e_min, e_max):
    m = (energy.to_value('TeV') > e_min.to_value('TeV')) & (energy.to_value('TeV') <= e_max.to_value('TeV'))
    return np.where(m, flux, 0) * flux.unit


def e_from_gamma(gamma):
    return ((gamma - 1) * constants.m_e * constants.c ** 2).to('TeV')


def meyer_model(
    a=8e47 / u.erg,
    alpha=2.5,
    e_min=1E10 * u.eV,
    e_max=5e15 * u.eV,
    beta=0.1,
):
    pwrl = LogParabola(amplitude=a, e_0=1 * u.TeV, alpha=alpha, beta=beta)
    f = lambda x: heaviside(x, pwrl(x), e_min, e_max) + 0.0001 / u.erg
    e = np.logspace(np.log10(e_min.to_value('TeV')), np.log10(e_max.to_value('TeV')), 400) * u.TeV
    return TableModel(e, f(e))



def ssc_model_components(parameters, precision=20):
    log_amplitude = parameters[0]
    alpha = parameters[1]
    beta = parameters[2]
    log_e_max = parameters[3]
    log_e_min = parameters[4]
    B = parameters[5]

    d_meyer = dict(a=(10**log_amplitude) / u.erg, alpha=alpha, beta=beta, e_max=10**log_e_max * u.eV, e_min=10**log_e_min * u.eV)

    T = meyer_model(**d_meyer)
    SYN = Synchrotron(T, B=B * u.uG, Eemax=50 * u.PeV, Eemin=0.01 * u.GeV, nEed=precision)

    # Compute photon density spectrum from synchrotron emission assuming R=2.1 pc
    Rpwn = 2.1 * u.pc
    Esy = np.logspace(-10, 10, 50) * u.MeV
    Lsy = SYN.flux(Esy, distance=0 * u.cm)  # use distance 0 to get luminosity
    phn_sy = Lsy / (4 * np.pi * Rpwn ** 2 * constants.c) * 2.25 # see section 1.6.2 in Ghiselini, Radiative Processes. 

    IC = InverseCompton(
        T,
        seed_photon_fields=[
            "CMB",
            ["FIR", 70 * u.K, 0.5 * u.eV / u.cm ** 3],
            ["NIR", 5000 * u.K, 1 * u.eV / u.cm ** 3],
            ["SSC", Esy, phn_sy],
        ],
        Eemax=50 * u.PeV,
        Eemin=0.01 * u.GeV,
        nEed=precision,
    )
    return SYN, IC


def ssc_model(pars, energy, precision=20):
    '''
    Adapted (very simplified) model of the Crab Nebula from Meyer et. al. 2010 
    
    returns the flux at given energy.
    
    Parameters
    ----------
    pars : list
        parameters [log(amp), alpha, beta, log_e_max, log_e_min, B]
    energy : array quantity
        energy at which toi evaluate the model
    
    Returns
    -------
    Quantity
        flux at given energies
    '''

    SYN, IC = ssc_model_components(pars, precision=precision)
    f = IC.flux(energy, distance=2 * u.kpc) + SYN.flux(energy, distance=2 * u.kpc)
    return f.to("cm-2 TeV-1 s-1")


@click.command()
@click.argument('output_path', type=click.Path(dir_okay=False))
@click.option('-N', '--sample_points', default=5, help='number of points to sample')
@click.option('-t', '--n_jobs', default=8, help='Number of jobs to start in parallel')
def create_lookup_table(output_path, sample_points=5, n_jobs=8):
    '''Creates a lookup table of model values for a simple self-synchrotron model using naima.
    The resulting lookup table is stored as a fits file under the OUTPUT_PATH.
    '''
    energy = np.logspace(-8, 3, 100) * u.TeV

    N = sample_points
    log_ampl = np.linspace(45.5, 48.5, N)
    alphas = np.linspace(2.5, 3.5, N)
    betas = np.linspace(0.0, 0.1, N)
    log_e_maxs = np.linspace(14.0, 16.2, N)
    log_e_mins = np.linspace(10.0, 12, N)
    bs = np.linspace(50, 150, N)

    parameters = list(product(log_ampl, alphas, betas, log_e_maxs, log_e_mins, bs))
    print(f'Calculating for {len(parameters)}')

    results = Parallel(n_jobs=n_jobs, verbose=1)(delayed(ssc_model)(p, energy) for p in parameters)
    r = np.array(results).T.reshape(len(energy), N, N, N, N, N, N)

    data_dict = {
        'energy': energy[np.newaxis, :],
        'log_ampl': log_ampl[np.newaxis, :],
        'alpha': alphas[np.newaxis, :],
        'beta': betas[np.newaxis, :],
        'log_e_max': log_e_maxs[np.newaxis, :],
        'log_e_min': log_e_mins[np.newaxis, :],
        'B': bs[np.newaxis, :] * u.uG,
        'data': r[np.newaxis, :]
    }

    t = Table(data_dict, meta={'INFO': 'Adapted Meyer SSC model evaluated on grid points'})
    print(f'Writing results to {output_path}')
    t.write(output_path, overwrite=True)



def ssc_model_lut(path='./lut/lut.fits'):
    t = Table.read(path)
    energy = t['energy'].data.ravel() * u.TeV
    log_energy = np.log10(energy.to_value(u.TeV))

    log_ampl = t['log_ampl'].data.ravel()
    alpha = t['alpha'].data.ravel()
    beta = t['beta'].data.ravel()
    log_e_max = t['log_e_max'].data.ravel()
    log_e_min = t['log_e_min'].data.ravel()
    B = t['B'].data.ravel()

    data = np.log10(t['data'].data.squeeze())

    f = RegularGridInterpolator((log_energy, log_ampl, alpha, beta, log_e_max, log_e_min, B), data, bounds_error=False, fill_value=None)

    def model(params, energy):
        FLUX_UNIT = u.Unit("cm-2 TeV-1 s-1")
        if np.isscalar(energy):
            return f([energy, *params]) * FLUX_UNIT

        es = energy.to_value(u.TeV)
        xs = [np.full_like(es, p) for p in params]
        xs = np.array([np.log10(es), *xs])
        return 10**f(xs.T) * FLUX_UNIT

    return model


if __name__ == "__main__":
    create_lookup_table()
