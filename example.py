import numpy as np
from physunits import *
from fit_traces import *
import emcee
import corner

π = np.pi

import matplotlib.pyplot as plt

plt.style.use("pacostyle")
use_real_data = False

cfig = plt.figure()
for N_samples in [8, 16, 32]:
    if not use_real_data:
        # True parameters
        wx0, wy0 = 314 * um, 137 * um
        zx0, zy0 = 62.7 * inch, 63.3 * inch
        lm = 1991 * nm

        zz = np.linspace(30 * inch, 105 * inch, N_samples)
        dz = 5 * mm
        np.random.seed(137)
        zz = np.random.normal(zz, dz)
        # Emulate data
        wu = wx0 * np.sqrt(1 + (zz - zx0) ** 2 / (π * wx0**2 / lm) ** 2)
        dw = 15 * um
        σ_wu = np.abs(dw * np.random.randn(wu.size))
        wu += dw * np.random.randn(wu.size)
        wv = wy0 * np.sqrt(1 + (zz - zy0) ** 2 / (π * wy0**2 / lm) ** 2)
        σ_wv = np.abs(dw * np.random.randn(wv.size))
        wv += dw * np.random.randn(wu.size)

    # Use real data
    if use_real_data:
        wu = (
            np.array(
                [
                    2035,
                    1800,
                    1300,
                    1011,
                    731,
                    400,
                    200,
                    420,
                    610,
                    960,
                    1309,
                    1680,
                    1990,
                ]
            )
            * um
            / 2
        )
        σ_wu = 1 * np.array([5, 10, 30, 25, 5, 5, 2, 5, 10, 5, 10, 15, 10]) * um

        wv = (
            np.array(
                [
                    1990,
                    1664,
                    1240,
                    900,
                    659,
                    378,
                    216,
                    441,
                    728,
                    1064,
                    1372,
                    1600,
                    1690,
                ]
            )
            * um
            / 2
        )
        σ_wv = 1 * np.array([5, 5, 10, 10, 5, 5, 2, 5, 10, 5, 10, 10, 15]) * um
        zz = np.linspace(52 * inch, 52 * inch + inch * len(wu), len(wu))
        σ_zz = np.random.normal(0, 3 * mm)
        zz += σ_zz

    xfit = Fit(x_data=zz, y_data=wu, u_data=σ_wu, model="waist")
    xfit.add_parameter(
        name="z0", value=67 * inch, vmin=50 * inch, vmax=200 * inch, vary=True
    )
    xfit.add_parameter(
        name="w0", value=195 / 2 * um, vmin=60 * um, vmax=1510 * um, vary=True
    )
    xfit.add_parameter(
        name="wavelength",
        value=1.95 * um,
        vmin=1.7 * um,
        vmax=2.2 * um,
        vary=True,
    )
    xfitpars, xcov, xfitbest, χ2x = xfit.fit(print_report=True)

    yfit = Fit(x_data=zz, y_data=wv, u_data=σ_wv, model="waist")
    yfit.add_parameter(
        name="z0", value=67 * inch, vmin=50 * inch, vmax=200 * inch, vary=True
    )
    yfit.add_parameter(
        name="w0", value=193 / 2 * um, vmin=60 * um, vmax=1510 * um, vary=True
    )
    yfit.add_parameter(
        name="wavelength",
        value=1.95 * um,
        vmin=1.7 * um,
        vmax=2.2 * um,
        vary=True,
    )
    yfitpars, ycov, yfitbest, χ2y = yfit.fit(print_report=True)

    z_extended = np.linspace(0.9 * zz[0], 1.1 * zz[-1], 2**8)
    xfitext = waist(z_extended, *xfitpars)
    yfitext = waist(z_extended, *yfitpars)
    yfiterr, yfiterr_extent = model_shaded_uncertainty(
        function=waist, x=z_extended, params=yfitpars, covariance=ycov
    )
    xfiterr, xfiterr_extent = model_shaded_uncertainty(
        function=waist, x=z_extended, params=xfitpars, covariance=xcov
    )

    # plt.figure(figsize=(9, 6))
    # ax = plt.subplot(211)
    # wuline, _, σ_wuline = ax.errorbar(
    #     zz, wu / um, yerr=σ_wu / um, xerr=2 * mm, fmt="", ecolor="r"
    # )
    # wvline, _, σ_wvline = ax.errorbar(
    #     zz, wv / um, yerr=σ_wv / um, xerr=2 * mm, fmt="", ecolor="b"
    # )
    # ax.imshow(
    #     np.log(yfiterr),
    #     extent=[
    #         yfiterr_extent[0],
    #         yfiterr_extent[1],
    #         yfiterr_extent[2] / um,
    #         yfiterr_extent[3] / um,
    #     ],
    #     cmap="Blues",
    #     aspect="auto",
    #     alpha=0.5,
    #     vmin=0,
    # )
    # ax.imshow(
    #     np.log(xfiterr),
    #     extent=[
    #         xfiterr_extent[0],
    #         xfiterr_extent[1],
    #         xfiterr_extent[2] / um,
    #         xfiterr_extent[3] / um,
    #     ],
    #     cmap="Reds",
    #     aspect="auto",
    #     alpha=0.5,
    #     vmin=0.0,
    # )

    # wuline.set_lw(0), wvline.set_lw(0)
    # σ_wuline[0].set_lw(3), σ_wvline[0].set_lw(3)
    # σ_wuline[1].set_lw(3), σ_wvline[1].set_lw(3)

    # wuline, _, σ_wuline = ax.errorbar(
    #     zz, -wu / um, yerr=σ_wu / um, xerr=2 * mm, fmt="", ecolor="r"
    # )
    # wvline, _, σ_wvline = ax.errorbar(
    #     zz, -wv / um, yerr=σ_wv / um, xerr=2 * mm, fmt="", ecolor="b"
    # )
    # ax.imshow(
    #     -np.log(yfiterr),
    #     extent=[
    #         yfiterr_extent[0],
    #         yfiterr_extent[1],
    #         -yfiterr_extent[2] / um,
    #         -yfiterr_extent[3] / um,
    #     ],
    #     cmap="Blues_r",
    #     aspect="auto",
    #     alpha=0.5,
    #     vmax=0,
    # )
    # ax.imshow(
    #     -np.log(xfiterr),
    #     extent=[
    #         xfiterr_extent[0],
    #         xfiterr_extent[1],
    #         -xfiterr_extent[2] / um,
    #         -xfiterr_extent[3] / um,
    #     ],
    #     cmap="Reds_r",
    #     aspect="auto",
    #     alpha=0.5,
    #     vmax=0,
    # )
    # wuline.set_lw(0), wvline.set_lw(0)
    # σ_wuline[0].set_lw(3), σ_wvline[0].set_lw(3)
    # σ_wuline[1].set_lw(3), σ_wvline[1].set_lw(3)

    # ax.scatter(zz, wu / um, color="crimson", marker="o", s=50, alpha=0.5)
    # ax.scatter(zz, -wu / um, color="crimson", marker="o", s=50, alpha=0.5)
    # ax.plot(
    #     z_extended,
    #     xfitext / um,
    #     label=Rf"$w_{{x}}(z_0)={xfitpars[1]/um:.0f}({np.sqrt(xcov[1, 1])/um:.0f})$ um at $z_0$={xfitpars[0]/inch:.2f}({np.sqrt(xcov[0,0])/inch*100:.0f})in",
    #     c="r",
    #     ls="--",
    #     lw=0.01,
    # )

    # ax.scatter(zz, wv / um, color="navy", marker="o", s=50, alpha=0.5)
    # ax.scatter(zz, -wv / um, color="navy", marker="o", s=50, alpha=0.5)
    # ax.plot(
    #     z_extended,
    #     yfitext / um,
    #     label=Rf"$w_{{y}}(z_0)={yfitpars[1]/um:.0f}({np.sqrt(ycov[1, 1])/um:.0f})$ um at $z_0$={yfitpars[0]/inch:.2f}({np.sqrt(ycov[0,0])/inch*100:.0f}) in",
    #     c="b",
    #     ls="--",
    #     lw=0.01,
    # )

    # ax.set_ylabel(Rf"beam radius (um)")
    # ax.legend(loc="upper center")
    # ax.set_ylim(-0, 1500)
    # ax.set_xlim(z_extended[0], z_extended[-1])

    # # Residuals
    # axr = plt.subplot(212, aspect=1 / 8.8e2, sharex=ax)
    # axr.set_ylabel(Rf"residuals (um)")
    # axr.set_xlabel(Rf"z (m)")
    # axr.scatter(
    #     zz,
    #     (wu - xfitbest) / um,
    #     marker="o",
    #     alpha=0.8,
    #     s=30,
    #     c="crimson",
    #     label=Rf"$\chi_\nu^{{2}}={χ2x:.4f}$",
    # )
    # axr.scatter(
    #     zz,
    #     (wv - yfitbest) / um,
    #     marker="o",
    #     alpha=0.8,
    #     s=30,
    #     c="navy",
    #     label=Rf"$\chi_\nu^{{2}}={χ2y:.4f}$",
    # )
    # axr.set_ylim(-50, 50)
    # axr.legend()
    # plt.subplots_adjust(wspace=-0.5, hspace=0)
    # plt.show()

    ## MCMC

    def lnprob(x, mu, icov):
        return -0.5 * np.dot((x - mu), np.dot(icov, (x - mu)))

    def mc_sens(
        x, modelfunc, fit_pars, fit_cov=None, n_points=5000, n_walkers=20
    ):
        # Initialize the emcee sampler, zero mean Gaussians with 1% stdev
        if fit_cov is None:
            std_jj = np.diag(0.01 * fit_pars)
            fit_cov = np.dot(std_jj, std_jj)
        p0 = np.random.rand(n_walkers, len(fit_pars))
        sampler = emcee.EnsembleSampler(
            n_walkers,
            len(fit_pars),
            lnprob,
            args=[fit_pars, np.linalg.inv(fit_cov)],
        )
        pos, prob, state = sampler.run_mcmc(p0, n_points)
        # Rerun around the likeliest sample distr.
        sampler.reset()
        _, _, _ = sampler.run_mcmc(pos, n_points, progress=True)
        return sampler.get_chain(flat=True)

    samples = mc_sens(
        z_extended, waist, yfitpars, ycov.T, n_points=1000, n_walkers=20
    )

    labels = [
        Rf"$z_0\,[\mathrm{{m}}]$",
        Rf"$w_0[\mathrm{{\mu m}}]$",
        Rf"$\lambda[\mathrm{{nm}}]$",
    ]
    units = list([m, um, nm])
    means = list([samples[:, jj].mean() for jj in range(len(yfitpars))])
    stds = list([samples[:, jj].std() for jj in range(len(yfitpars))])
    ranges = list(
        [
            ((mn - 5 * st) / un, (mn + 5 * st) / un)
            for mn, st, un in zip(means, stds, units)
        ]
    )

    scaled_samples = np.zeros_like(samples)
    for jj in range(len(yfitpars)):
        scaled_samples[:, jj] = samples[:, jj] / units[jj]

    cfig = corner.corner(
        scaled_samples,
        fig=cfig,
        labels=labels,
        levels=(0.95,),
        color=f"C{int(np.log2(N_samples))}",
        bins=50,
        # range=ranges,
        # show_titles=True,
        hist_kwargs={"linewidth": 2, "alpha": 0.7, "rasterized": True},
        hist2d_kwargs={"rasterized": True},
        label_kwargs={"fontsize": 16, "fontweight": "bold"},
        title_kwargs={"fontsize": 16, "fontweight": "bold"},
    )

    txt = ""
    for jj in range(3):
        mcmc = np.percentile(scaled_samples[:, jj], [16, 50, 84, 95, 16])
        q = np.diff(mcmc)
        txt += (
            Rf"{labels[jj]}$ = {mcmc[1]:.3f}_{{-{q[0]:.3f}}}^{{+{q[1]:.3f}}}$, "
        )

    corner.overplot_lines(
        cfig,
        yfitpars,
        reverse=False,
        label=f"N={N_samples}",
        color=f"C{int(np.log2(N_samples))}",
    )
corner.overplot_lines(
    cfig, [zy0 / m, wy0 / um, lm / nm], reverse=False, label=f"Truth", color="k"
)
plt.legend()
# fig.suptitle(txt)
plt.show()
