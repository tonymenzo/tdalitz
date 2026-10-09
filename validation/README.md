# Validation

Each reference channel was checked against the exact expectations computed from the amplitude table (not fits). The figures in [`figures/`](figures/) show these comparisons.

| check | result |
| --- | --- |
| amplitude tables against Laura++ (50k points per channel, $A$ and $\bar A$) | agree to $\leq 4\times10^{-14}$, i.e. to rounding |
| Dalitz maps, mass projections, decay-time distributions and tag asymmetries (51 $\chi^2$ comparisons over the four channels) | $p$-values consistent with uniform: Kolmogorov–Smirnov $p = 0.02$, $0.37$, $0.44$ and $0.84$ for four independent seeds |
| decay-time resolution ($B_s^0$, $\sigma_t = 45$ fs, test phase $\arg(q/p) = -0.4$) | measured damping $0.728 \pm 0.016$, expected $0.726$ |
| mistag ($B^0$, $\omega = 0.2$) | asymmetry dilution as expected ($p = 0.47$) |

Figures per channel (`d0`, `b0`, `bs`, `b0_3pi`):

* `<channel>_amplitude.pdf`: table against Laura++, and realized against target fit fractions;
* `<channel>_dalitz_maps.pdf`: generated Dalitz plot and pulls for each tag;
* `<channel>_dalitz_masses.pdf`: the three invariant-mass projections for each tag;
* `<channel>_time.pdf`: decay time ($\Delta t$ for `b0_3pi`) for each tag, and the tag asymmetry in Dalitz-plot regions;
* `detector.pdf`: mistag and time-resolution effects.
