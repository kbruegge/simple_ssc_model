import matplotlib as mpl
from naima import uniform_prior
import naima
import numpy as np
import astropy.units as u
from data import read_crab_mwl_data
from astropy.table import Table
from scipy.interpolate import RegularGridInterpolator
import click
import os


FLUX_UNIT = u.Unit("cm-2 TeV-1 s-1")


def lut_model(params, energy):
    if np.isscalar(energy):
        return f([energy, *params]) * FLUX_UNIT

    es = energy['energy'].to_value(u.TeV)
    xs = [np.full_like(es, p) for p in params]
    xs = np.array([np.log10(es), *xs])
    return 10**f(xs.T) * FLUX_UNIT


def lut_prior(pars):
    lnprior = (
        uniform_prior(pars[0],   log_ampl[0],  log_ampl[-1])
        + uniform_prior(pars[1], alpha[0],     alpha[-1])
        + uniform_prior(pars[2], beta[0],      beta[-1])
        + uniform_prior(pars[3], log_e_max[0], log_e_max[-1])
        + uniform_prior(pars[4], log_e_min[0], log_e_min[-1])
        + uniform_prior(pars[5], B[0],         B[-1])
    )
    return lnprior


@click.command()
@click.argument('input_path', type=click.Path(dir_okay=False))
@click.argument('output_path', type=click.Path(dir_okay=True, file_okay=False))
@click.option('-t', '--n_job', default=4, help='Number of jobs to start in parallel')
@click.option('-s', '--n_sample', default=1500, help='Number of samples in chain')
@click.option('-s', '--n_burn', default=400, help='Number of burn in samples')
@click.option('-s', '--n_walker', default=300, help='Number of MCMC walkers')
def main(input_path, output_path, n_job, n_sample, n_burn, n_walker):

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # but kai?? why you doing global things?
    # because i have to. for the sake of speed. And for queen and country of course.
    # see https://emcee.readthedocs.io/en/latest/tutorials/parallel/#parallel
    global f
    global B
    global data
    global log_ampl
    global alpha
    global beta 
    global log_e_min
    global log_e_max



    labels = ["log_main_amplitude", "alpha", "beta", 'log10(E_max)', 'log10(E_min)', 'B']
    p0 = np.array([48, 3, 0.05, 15.6, 11, 100])

    t = Table.read(input_path)
    energy = t['energy'].data.ravel() * u.TeV
    log_energy = np.log10(energy.to_value(u.TeV))

    log_ampl = t['log_ampl'].data.ravel()
    alpha = t['alpha'].data.ravel()
    beta = t['beta'].data.ravel()
    log_e_max = t['log_e_max'].data.ravel()
    log_e_min = t['log_e_min'].data.ravel()
    B = t['B'].data.ravel()

    print('log ampl', log_ampl)
    print('alpha', alpha)
    print('beta', beta)
    print('e_max', log_e_max)
    print('e_min', log_e_min)
    print('B', B)

    data = np.log10(t['data'].data.squeeze())

    f = RegularGridInterpolator((log_energy, log_ampl, alpha, beta, log_e_max, log_e_min, B), data, bounds_error=False, fill_value=None)


    data = read_crab_mwl_data(e_min=40 * u.keV)
    sampler, pos = naima.run_sampler(
        data_table=data,
        p0=p0,
        labels=labels,
        model=lut_model,
        prior=lut_prior,
        nwalkers=n_walker,
        nburn=n_burn,
        nrun=n_sample,
        threads=n_job,
        prefit=True,
    )

    mpl.use('Agg')
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams['backend'] = 'agg'

    naima.save_run(f"{output_path}/crab_chain.h5", sampler, clobber=True)
    naima.save_results_table(f"{output_path}/crab_naima_fit", sampler)
    naima.save_diagnostic_plots(f"{output_path}/crab_naima", sampler)


if __name__ == "__main__":
    main()
