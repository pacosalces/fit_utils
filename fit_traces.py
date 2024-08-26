import numpy as np
from lmfit import Parameters, minimize, report_fit
import corner

from scipy import optimize
from scipy import stats


class Fit(object):

    def __init__(
        self, x_data, y_data, u_data, model, mcmc=False, *args, **kwargs
    ):
        self.x_data = x_data
        if "z_data" in kwargs.keys():
            self.y_data = y_data
            self.fit_data = kwargs["z_data"]
            self.u_fit_data = u_data
            self.data_pack = {
                "x": self.x_data,
                "y": self.y_data,
                "z": self.fit_data,
                "uz": self.u_fit_data,
            }
        else:
            self.fit_data = y_data
            self.u_fit_data = u_data
            self.data_pack = {
                "x": self.x_data,
                "y": None,
                "z": self.fit_data,
                "uz": self.u_fit_data,
            }
        self.model = model
        self.mcmc = mcmc
        self.pars = Parameters()
        self.pars_pack = []

    def add_parameter(self, name, value, vmin, vmax, vary):
        self.pars.add(name=name, value=value, min=vmin, max=vmax, vary=vary)

    def fit(self, print_report=False):
        if self.data_pack["y"] is not None:
            self.independent = [self.data_pack["x"], self.data_pack["y"]]
        else:
            self.independent = [self.data_pack["x"]]

        def residuals(parameters):
            self.pars_pack = parameters.valuesdict().values()

            model_command = self.model + "(*self.independent, *self.pars_pack)"
            globals()["evaluated_model"] = eval(model_command)
            return (self.data_pack["z"] - evaluated_model) / self.data_pack[
                "uz"
            ]

        result = minimize(
            residuals, self.pars, method="leastsq", nan_policy="omit"
        )
        if self.mcmc:
            result.params.add(
                "__lnsigma", value=np.log(0.1), min=np.log(0.001), max=np.log(2)
            )
            mcres = minimize(
                residuals,
                method="emcee",
                nan_policy="omit",
                burn=300,
                steps=1000,
                thin=20,
                params=result.params,
                is_weighted=False,
                progress=False,
            )

            emcee_plot = corner.corner(
                mcres.flatchain,
                labels=mcres.var_names,
                truths=list(mcres.params.valuesdict().values()),
            )

        redchisq = result.redchi
        optimum_pars = np.array(
            [result.params[key].value for key in result.params.keys()]
        )
        covariance_matrix = np.array(result.covar)
        optimum_result = eval(
            self.model + "(*self.independent, *self.pars_pack)"
        )
        if print_report:
            report_fit(result)
        return optimum_pars, covariance_matrix, optimum_result, redchisq


def partial_derivatives(function, x, params, u_params):
    model_at_center = function(x, *params)
    partial_derivatives = []
    for i, (param, u_param) in enumerate(zip(params, u_params)):
        d_param = u_param / 1e6
        params_with_partial_differential = np.zeros(len(params))
        params_with_partial_differential[:] = params[:]
        params_with_partial_differential[i] = param + d_param
        model_at_partial_differential = function(
            x, *params_with_partial_differential
        )
        partial_derivative = (
            model_at_partial_differential - model_at_center
        ) / d_param
        partial_derivatives.append(partial_derivative)
    return partial_derivatives


def model_uncertainty(function, x, params, covariance):
    u_params = [np.sqrt(np.abs(covariance[i, i])) for i in range(len(params))]
    model_partial_derivatives = partial_derivatives(
        function, x, params, u_params
    )
    try:
        squared_model_uncertainty = np.zeros(x.shape)
    except TypeError:
        squared_model_uncertainty = 0
    for i in range(len(params)):
        for j in range(len(params)):
            squared_model_uncertainty += (
                model_partial_derivatives[i]
                * model_partial_derivatives[j]
                * covariance[i, j]
            )
    return np.sqrt(squared_model_uncertainty)


def model_shaded_uncertainty(
    function,
    x,
    params,
    covariance,
    yrange=None,
    resolution=1024,
    columns_normalised=False,
):
    model_mean = function(x, *params)
    model_stddev = model_uncertainty(function, x, params, covariance)
    if yrange is None:
        yrange = [
            (model_mean - 10 * model_stddev).min(),
            (model_mean + 10 * model_stddev).max(),
        ]
    y = np.linspace(yrange[0], yrange[1], resolution)
    Model_Mean, Y = np.meshgrid(model_mean, y)
    Model_Stddev, Y = np.meshgrid(model_stddev, y)
    if columns_normalised:
        probability = np.exp(-((Y - Model_Mean) ** 2) / (2 * Model_Stddev**2))
    else:
        probability = np.normpdf(Y, Model_Mean, Model_Stddev)
    #    probability = (abs(Y - Model_Mean)/Model_Stddev).clip(0,3)
    return probability, [x.min(), x.max(), y.min(), y.max()]


def linear(x, m, c):
    return m * x + c


def quadratic(x, a, b, c):
    return a * x**2 + b * x + c


def stepfn(t, initial, final, step_time, rise_time):
    # 10-90% rise time, approx sqrt(2)*log(9)/5 ~= 0.62 x (1/e^2 integration width):
    exponent = -4 * np.log(3) * (t - step_time) / rise_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent, 700])
    return (final - initial) / (1 + np.exp(exponent)) + initial


def step_fermi(x, min, max, width, delay):
    return (max - min) * (1 / (1 + np.exp(-(x - delay) / (width)))) - min


def lorentzian(x, amplitude, x0, fwhm, offset):
    return amplitude / (1 + (x - x0) ** 2 / (fwhm / 2) ** 2) + offset


def square_root(x, x0, a, b):
    return a * np.sqrt((x - x0)) + b


def multi_lorentzian(x, *pars):
    n_peaks = 2
    n_pars = np.size(pars)
    step = np.int(4 / n_peaks)
    amps, x0s, widths, offsets = (
        pars[0:step],
        pars[step : 2 * step],
        pars[2 * step : 3 * step],
        pars[3 * step : 4 * step],
    )
    first_peak = lorentzian(x, amps[0], x0s[0], widths[0], offsets[0])
    peaks = np.array(first_peak)
    for each in range(1, n_peaks):
        a, x0, w, off = amps[each], x0s[each], widths[each], offsets[each]
        peaks = peaks + np.array(lorentzian(x, a, x0, w, off)) / n_peaks
    return np.array(lorentzian(x, a, x0, w, off))


def gaussian(x, amplitude, x0, sigma, offset):
    x = np.array(x)
    return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


def gaussian_2d(x, y, theta, x0, y0, sigma_x, sigma_y, amplitude, offset):
    theta *= np.pi / 180
    u = (x - x0) * np.cos(theta) - (y - y0) * np.sin(theta)
    v = (x - x0) * np.sin(theta) + (y - y0) * np.cos(theta)
    return (
        amplitude * np.exp(-0.5 * (u / sigma_x) ** 2 - 0.5 * (v / sigma_y) ** 2)
        + offset
    )


def laguerre_gaussian_2d(x, y, x0, y0, sigma_x, sigma_y, amplitude, offset):
    u = (x - x0) / (sigma_x)
    v = (y - y0) / (sigma_y)
    return amplitude * (2 * (u**2 + v**2)) * np.exp(-2 * (u**2 + v**2)) + offset


def thomas_fermi(x, amplitude, x0, xTF, offset):
    if iterable(x):
        return np.array(
            [
                amplitude * max(1 - (xi - x0) ** 2 / xTF**2, 0) ** (3 / 2)
                + offset
                for xi in x
            ]
        )
    else:
        return (
            amplitude * max(1 - (x - x0) ** 2 / xTF**2, 0) ** (3 / 2) + offset
        )


def bimodal(x, amp1, amp2, x0, sigma, xTF, offset):
    if iterable(x):
        return np.array(
            [
                amp1 * np.exp(-((xi - x0) ** 2) / (2 * sigma**2))
                + amp2 * max(1 - (xi - x0) ** 2 / xTF**2, 0) ** (3 / 2)
                + offset
                for xi in x
            ]
        )
    else:
        x = np.array(x)
        return (
            amp1 * np.exp(-((x - x0) ** 2) / (2 * sigma**2))
            + amp2 * max(1 - (x - x0) ** 2 / xTF**2, 0) ** (3 / 2)
            + offset
        )


def expfn(t, initial, final, decay_time, zero_time=False):
    t = np.array(t)
    if zero_time:
        exponent = -(t - np.amin(t)) / decay_time
    else:
        exponent = -t / decay_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent, 700])
    return (final - initial) * (1 - np.exp(exponent)) + initial


def log_expfn(t, initial, final, decay_time):
    t = np.array(t)
    exponent = -t / decay_time
    # make it immune to overflow errors, without sacrificing accuracy:
    if iterable(exponent):
        exponent[exponent > 700] = 700
    else:
        exponent = min([exponent, 700])
    return np.log((final - initial) * (1 - np.exp(exponent)) + initial)


def exp_initial_rate(
    initial, final, decay_time, d_initial, d_final, d_decay_time
):
    rate = (final - initial) / decay_time
    d_rate = rate * np.sqrt(
        (d_initial**2 + d_final**2) / (final - initial) ** 2
        + d_decay_time**2 / decay_time**2
    )
    return rate, d_rate


def rabi(t, f, f0, fR, A=1, c=0):
    t = np.array(t)
    det = f - f0
    Fz = -(
        det**2 + fR**2 * np.cos(2 * np.pi * t * np.sqrt(det**2 + fR**2))
    ) / (det**2 + fR**2)
    return A * Fz + c


def sine(t, f, A, c, phi):
    t = np.array(t)
    return A * np.sin(2 * np.pi * f * t + phi) + c


def cos_ramsey(phi, A=1, c=0, dphi=0):
    # Note: phi is in degrees
    return A * np.cos(np.pi * phi / 180 + dphi) + c


def line_noise(t, amps, phases, offset=0, gradient=0):
    freqs = 50 * (np.arange(len(amps)) + 1)
    harmonics = np.array(
        [
            A * np.sin(2 * np.pi * f * t + phi)
            for (A, f, phi) in zip(amps, freqs, phases)
        ]
    )
    return harmonics.sum(axis=0) + offset + gradient * t


def sine_decay(t, f, A=1, c=0, phi=0, tc=1e1):
    t = np.array(t)
    return A * np.exp(-t / tc) * np.sin(2 * np.pi * f * t + phi) + c


def ramsey(t, q, A, f, tc, c=0, phi=0):
    # return -A*exp(-t/tc)*cos(2*pi*f*t+phi)+c
    return (
        -A
        * np.exp(-(t**2) / tc**2)
        * np.cos(2 * np.pi * q * t)
        * np.cos(2 * np.pi * f * t + phi)
        + c
    )


def waist(z, z0, w0, wavelength, Msquared):
    z = z - z0
    zR = np.pi * (Msquared * w0) ** 2 / wavelength
    return w0 * np.sqrt(1 + (z / zR) ** 2)
