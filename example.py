"""example.py -- PHYSICAL induced shield, example-style two-panel figure.Coil currents solved for the centre quadrupole;
 shield currents are theeddy currents a real copper shell actually induces (multipole moments xthin-shell L/R response, see shield_common.induced_shield_currents)
.Same layout as shield2D.py so the two methods are directly comparable.Run:  python example.py"""import numpy as npfrom shield_common import (make_template, solve_quad_coils, induced_shield_currents,                           build_pair, leakage_report, two_panel)

main()
:    tpl = make_template()
    I_coil = solve_quad_coils(tpl)
    I_shield, info = induced_shield_currents(tpl, I_coil)
    print(f"skin depth {info['delta']*1e6:.1f} um (d_eff {info['d_eff']*1e6:.1f} um)
;
  "          f"m=2 shielding factor {info['S2']:.3e}")
    print(f"coil current: max {np.max(np.abs(I_coil)
)
:.1f} A;
  "          f"induced shield current: peak {np.max(np.abs(I_shield)
)
:.1f} A")
    shielded, unshielded = build_pair(tpl, I_coil, I_shield)
    leakage_report(shielded, unshielded)
    two_panel(shielded, unshielded,              "physical induced shield", "figures/induced_shield_field.png")

__name__ == "__main__":    main()
