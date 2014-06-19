import numpy as np

def mh_hp(m, hpdfs, hcondpdfs, hsamplers):
    """
    run one iteration of metropolis hastings for hyperparameter estimation
    """
    # XXX: this can be done in parallel
    for fi, (hpdf, hcondpdf, hsamp) in enumerate(zip(hpdfs, hcondpdfs, hsamplers)):
        # this duplicates code from below, but avoids some overhead
        # by mutating state directly
        cur_hp = m.get_feature_hp(fi)
        prop_hp = hsamp(cur_hp)
        lg_pcur = hpdf(cur_hp) + m.score_data(fi)
        m.set_feature_hp(fi, prop_hp)
        lg_pstar = hpdf(prop_hp) + m.score_data(fi)
        lg_qbackwards = hcondpdf(prop_hp, cur_hp)
        lg_qforwards = hcondpdf(cur_hp, prop_hp)
        alpha = lg_pstar + lg_qbackwards - lg_pcur - lg_qforwards
        if alpha <= 0.0 and np.random.random() >= np.exp(alpha):
            # reject
            m.set_feature_hp(fi, cur_hp)

def mh_sample(xt, pdf, condpdf, condsamp):
    # sample xprop ~ Q(.|xt)
    xprop = condsamp(xt)

    # compute acceptance probability
    lg_alpha_1 = pdf(xprop) - pdf(xt)
    lg_alpha_2 = condpdf(xprop, xt) - condpdf(xt, xprop)
    lg_alpha   = lg_alpha_1 + lg_alpha_2

    # accept w.p. alpha(xprop, xt)
    if lg_alpha >= 0.0 or np.random.random() <= np.exp(lg_alpha):
        return xprop
    else:
        return xt
