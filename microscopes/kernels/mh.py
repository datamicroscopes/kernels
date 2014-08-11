import numpy as np


def sample(xt, pdf, condpdf, condsamp):
    # sample xprop ~ Q(.|xt)
    xprop = condsamp(xt)

    # compute acceptance probability
    lg_alpha_1 = pdf(xprop) - pdf(xt)
    lg_alpha_2 = condpdf(xprop, xt) - condpdf(xt, xprop)
    lg_alpha = lg_alpha_1 + lg_alpha_2

    # accept w.p. alpha(xprop, xt)
    if lg_alpha >= 0.0 or np.random.random() <= np.exp(lg_alpha):
        return xprop
    else:
        return xt
