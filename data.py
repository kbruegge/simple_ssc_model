from astropy.table import Table


def read_crab_mwl_data(component='nebula', e_min=None, path='./crab_mwl.fits.gz', paper=None):
    t = Table.read(path)
    if component:
        m = t['component'] == component
        t = t[m]

    if e_min:
        m = t['energy'].quantity.to_value('keV') > e_min.to_value('keV')
        t = t[m]

    if paper:
        m = t['paper'] == paper
        t = t[m]

    energy = t['energy'].quantity.to('TeV')
    flux = (t['energy_flux'].quantity * energy ** (-2)).to('TeV-1 cm-2 s-1')
    flux_error_lo = (t['energy_flux_err_lo'].quantity * energy ** (-2)).to('TeV-1 cm-2 s-1')
    flux_error_hi = (t['energy_flux_err_hi'].quantity * energy ** (-2)).to('TeV-1 cm-2 s-1')
    data_table = Table(
        {'energy': energy, 'flux': flux, 'flux_error_lo': flux_error_lo, 'flux_error_hi': flux_error_hi, 'paper': t['paper']}
    )
    data_table.sort('energy')
    return data_table
