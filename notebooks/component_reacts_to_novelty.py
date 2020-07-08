from scipy import optimize
import numpy


def exponential_decay_model(data, t_step=20.0):
    """
    data: trials x time
    """
    initial_value = data[:, 0].mean()
    func = lambda v, t: initial_value * numpy.exp(-t / v)
    t_vals = t_step * numpy.arange(data.shape[1])
    t_vals = t_vals.reshape((1, len(t_vals)))
    
    error_fun = lambda v, t: numpy.sum((func(v, t) - data) ** 2)
    res = optimize.fmin(error_fun, 40.0, args=(t_vals,), disp=False)
    v_val = res[0]
    return func(v_val, t_vals)


def reacts_to_novelty_model(traces, n=2):
    mdl_vals = exponential_decay_model(traces)
    var_before = numpy.var(traces, axis=1).mean()
    var_after = numpy.var(traces - mdl_vals, axis=1).mean()
    return (var_before - var_after) / var_before


def eval_reacts_to_novelty_model(traces_for_stims, **kwargs):
    return reacts_to_novelty_model(numpy.hstack(traces_for_stims).transpose())

