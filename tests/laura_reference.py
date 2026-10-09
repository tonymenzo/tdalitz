"""Total amplitudes evaluated directly by Laura++, as an independent reference.

This builds a ``LauIsobarDynamics`` with the *full* coefficient set and asks
Laura++ for the summed amplitude, so it shares no code path with the exporter
(which evaluates one unit-coefficient component at a time).
"""

import numpy as np


def laura_total(model, coefficients, m13, m23, library):
    """Laura++'s total amplitude ``sum_r c_r F_r`` at ``(m13^2, m23^2)``.

    ``coefficients`` maps component name to a complex number.
    """
    import ROOT
    ROOT.gSystem.Load(library)
    daughters = ROOT.LauDaughters(model.parent, *model.daughters, False)
    vetoes = ROOT.LauVetoes()                     # Laura++ keeps raw pointers:
    eff = ROOT.LauEffModel(daughters, vetoes)     # keep every object alive here
    maker = ROOT.LauResonanceMaker.get()
    for cat, r in model.barrier_radii.items():
        maker.setDefaultBWRadius(getattr(ROOT.LauBlattWeisskopfFactor, cat), r)
    dyn = ROOT.LauIsobarDynamics(daughters, eff)
    for name, bachelor in model.resonances:
        res = dyn.addResonance(name, bachelor,
                               getattr(ROOT.LauAbsResonance, model.lineshapes.get(name, "RelBW")))
        if name in model.mass_width_overrides:
            res.changeResonance(*model.mass_width_overrides[name])
    vec = ROOT.std.vector("LauComplex")()
    for name, _ in model.resonances:
        c = complex(coefficients[name])
        vec.push_back(ROOT.LauComplex(c.real, c.imag))
    dyn.initialise(vec)
    out = np.empty(len(m13), dtype=complex)
    for i in range(len(m13)):
        dyn.calcLikelihoodInfo(float(m13[i]), float(m23[i]))
        a = dyn.getEvtDPAmp()
        out[i] = complex(a.re(), a.im())
    return out
