__copyright__ = """Copyright (C) 2023 George N. Wong"""
__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import subprocess
import numpy as np
from scipy.optimize import brentq


def evaluate_flux_difference(munit, dumpfiles, target_flux, logname=None, unpol=False,
                             **kwargs):
    """
    Evaluate the difference between the target flux and the flux from a set of dump
    files given a value of munit.

    :arg munit: value of munit to use

    :arg dumpfiles: list of dump files to use

    :arg target_flux: target flux to compare against

    :arg logname: (default=None) name of log file to write to (None for no log file)

    :arg unpol: (default=False) run in unpolarized "total intensity" mode

    :arg kwargs: keyword arguments to pass to get_fluxes(...) (e.g., rlow, thetacam, etc.)

    :returns: difference between the target flux and the flux from the dump files
    """
    Ftots = []

    for dumpfile in dumpfiles:
        try:
            Ftot_unpol, Ftot = get_fluxes(dumpfile, munit=munit, unpol=unpol, **kwargs)
            if unpol:
                Ftots.append(Ftot_unpol)
            else:
                Ftots.append(Ftot)
        except Exception as e:
            print(f"! error processing {dumpfile}: {e}")

    Ftot = np.array(Ftots).mean()
    print(f"tried {munit} and got {Ftot} (target {target_flux})")
    if logname is not None:
        fp = open(logname, 'a')
        fp.write(f"iteration {munit} -> {Ftot} (target {target_flux})\n")
        fp.close()
    return Ftot - target_flux


def get_seed_value(dumpfiles, target_flux, munit_low, munit_high, logname=None,
                   xtol=0.05, **kwargs):
    """
    Step through range of seed values to find a good starting point for the flux fitting.

    :arg dumpfiles: list of dump files to use

    :arg target_flux: target flux to compare against

    :arg munit_low: lower bound on munit

    :arg munit_high: upper bound on munit

    :arg logname: (default=None) name of log file to write to (None for no log file)

    :arg xtol: (default=0.05) x tolerance for the root finding algorithm

    :arg kwargs: keyword arguments to pass to get_fluxes(...) (e.g., rlow, thetacam, etc.)

    :returns: seed value for the flux fitting
    """

    flux_differences = []
    munits = np.logspace(np.log10(munit_low), np.log10(munit_high), 11)
    munit_low = None
    munit_high = None
    for mi, munit in enumerate(munits):
        flux_differences.append(evaluate_flux_difference(munit, dumpfiles, target_flux,
                                                         logname=logname, **kwargs))
        if flux_differences[-1] > 0:
            if mi > 0:
                munit_high = munit
            else:
                munit_high = munit
                munit_low = munit / 10.
            break
        munit_low = munit
    precise_text = f">> starting more precise fit with ({munit_low}, {munit_high})"
    print(precise_text)
    if logname is not None:
        fp = open(logname, 'a')
        fp.write(precise_text + "\n")
        fp.close()
    return fit_munit(dumpfiles, target_flux, munit_low, munit_high, logname=logname,
                     xtol=xtol, **kwargs)


def fit_munit(dumpfiles, target_flux, munit_low, munit_high, logname=None, xtol=None,
              fit_as_log=False, **kwargs):
    """
    Find the value of munit that gives the target flux.

    :arg dumpfiles: list of dump files to use

    :arg target_flux: target flux to compare against

    :arg munit_low: lower bound on munit

    :arg munit_high: upper bound on munit

    """

    if xtol is None:
        xtol = munit_low/10.

    # wrapper functions to allow fitting in log space
    if fit_as_log:
        def fa(x):
            return np.exp(x)
        def fb(x):  # noqa: E306
            return np.log(x)
    else:
        def fa(x):
            return x
        def fb(x):  # noqa: E306
            return x

    root = brentq(lambda x: evaluate_flux_difference(fa(x), dumpfiles, target_flux,
                  logname=logname, **kwargs), fb(munit_low), fb(munit_high), xtol=xtol)
    munit = fa(root)

    if logname is not None:
        fp = open(logname, 'a')
        fp.write(f"result {munit}\n\n")
        fp.close()

    return munit


def run_ipole(dumpfile, outfile=None, rlow=1, rhigh=40, thetacam=163, target='m87',
              munit=1.e25, freqcgs=230.e9, res=160, verbose=False, unpol=False,
              tracef=None, executable="./ipole", onlyargs=False):
    """
    Wrapper for ipole executable.

    :arg dumpfile: input dump file

    :arg outfile: (default=None) output file (None for no output file)

    :arg rlow: (default=1) r_low electron temperature prescription parameter

    :arg rhigh: (default=40) r_high electron temperature prescription parameter

    :arg thetacam: (default=163) camera inclination in degrees

    :arg target: (default='m87') target source for which to use default parameters

    :arg munit: (default=1.e25) mass unit for the simulation

    :arg freqcgs: (default=230.e9) observing frequency in cgs units

    :arg res: (default=160) resolution of the image per side in pixels

    :arg verbose: (default=False) print the command that is being run

    :arg unpol: (default=False) run in unpolarized "total intensity" mode

    :arg tracef: (default=None) trace file to write to (None for no trace file)

    :arg executable: (default="./ipole") path to the ipole executable

    :arg onlyargs: (default=False) return the arguments instead of running the executable

    :returns: list of strings containing the output of the executable
    """

    if target is None:
        target = "m87"
    target = target.lower()

    if target == "m87":
        mbh = 6.5e9
        dsource = 16.8e6
    elif target == "sgra":
        mbh = 4.1e6
        dsource = 8127
    else:
        print(f"! unrecognized target \"{target}\"")

    freqarg = f"--freqcgs={freqcgs}"
    mbharg = f"--MBH={mbh}"
    munitarg = f"--M_unit={munit}"
    dsourcearg = f"--dsource={dsource}"
    incarg = f"--thetacam={thetacam}"
    rlowarg = f"--trat_small={rlow}"
    rhigharg = f"--trat_large={rhigh}"
    dumpfilearg = f"--dump={dumpfile}"
    resarg = f"--nx={res} --ny={res}"

    args = [executable, freqarg, mbharg, munitarg, dsourcearg, incarg, rlowarg,
            rhigharg, dumpfilearg, resarg]

    if tracef is not None:
        args += [f"--trace_outf={tracef}"]

    if outfile is None:
        args += ["-quench"]
    else:
        args += [f"--outfile={outfile}"]

    if unpol:
        args += ["-unpol"]

    if onlyargs:
        return args

    if verbose:
        print(" ... running \"" + " ".join(args) + "\"")

    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = [z for y in [str(x)[2:-1].split('\\n') for x in p.communicate()] for z in y]

    return output


def get_fluxes(dumpfile, rlow=1, rhigh=40, thetacam=163, target='m87', munit=1.e25,
               freqcgs=230.e9, res=160, verbose=False, unpol=False):
    """
    Get the fluxes from an ipole run.

    :arg dumpfile: input dump file

    :arg rlow: (default=1) r_low electron temperature prescription parameter

    :arg rhigh: (default=40) r_high electron temperature prescription parameter

    :arg thetacam: (default=163) camera inclination in degrees

    :arg target: (default='m87') target source for which to use default parameters

    :arg munit: (default=1.e25) mass unit for the simulation

    :arg freqcgs: (default=230.e9) observing frequency in cgs units

    :arg res: (default=160) resolution of the image per side in pixels

    :arg verbose: (default=False) print the command that is being run

    :arg unpol: (default=False) run in unpolarized "total intensity" mode

    :returns: tuple of the unpolarized and polarized fluxes
    """

    exe = "./ipole"

    target = target.lower()

    if target == 'm87':
        mbh = 6.5e9
        dsource = 16.8e6
    elif target == 'sgra':
        mbh = 4.1e6
        dsource = 8127
    else:
        print(f"! unrecognized target \"{target}\"")

    freqarg = f"--freqcgs={freqcgs}"
    mbharg = f"--MBH={mbh}"
    munitarg = f"--M_unit={munit}"
    dsourcearg = f"--dsource={dsource}"
    incarg = f"--thetacam={thetacam}"
    rlowarg = f"--trat_small={rlow}"
    rhigharg = f"--trat_large={rhigh}"
    dumpfilearg = f"--dump={dumpfile}"
    resarg = f"--nx={res} --ny={res}"

    args = [exe, freqarg, mbharg, munitarg, dsourcearg, incarg, rlowarg,
            rhigharg, dumpfilearg, resarg]
    args += ["-quench"]
    if unpol:
        args += ["-unpol"]

    if verbose:
        print(" ... running \"" + " ".join(args) + "\"")

    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = [z for y in
              [str(x)[2:-1].split('\\n') for x in proc.communicate()] for z in y]
    Ftot_line = [line for line in output if 'unpol xfer' in line][0]
    st = Ftot_line.split()
    Ftot_unpol = float(st[-2+st.index('unpol')][1:])
    Ftot = float(st[-4+st.index('unpol')])

    return Ftot_unpol, Ftot
