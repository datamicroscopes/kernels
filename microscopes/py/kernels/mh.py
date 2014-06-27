import numpy as np

def mh_hp(m, hparams):
    """
    hparams: dict mapping feature id to the following dict:
        {
            'hpdf' : <a function taking values returned by `hsampler' below and returning scores>,
            'hcondpdf': <a 2-arg function TODO: fill me in>
            'hsamp': <TODO: fill me in>
        }
    """
    # XXX: this can be done in parallel
    for fi, hparam in hparams.iteritems():
        hpdf, hcondpdf, hsamp = hparam['hpdf'], hparam['hcondpdf'], hparam['hsamp']
        # this duplicates code from below, but avoids some overhead
        # by mutating state directly
        cur_hp = m.get_feature_hp_raw(fi)
        prop_hp = hsamp(cur_hp)
        lg_pcur = hpdf(cur_hp) + m.score_data(fi)
        m.set_feature_hp_raw(fi, prop_hp)
        lg_pstar = hpdf(prop_hp) + m.score_data(fi)
        lg_qbackwards = hcondpdf(prop_hp, cur_hp)
        lg_qforwards = hcondpdf(cur_hp, prop_hp)
        alpha = lg_pstar + lg_qbackwards - lg_pcur - lg_qforwards
        if alpha <= 0.0 and np.random.random() >= np.exp(alpha):
            # reject
            m.set_feature_hp_raw(fi, cur_hp)

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
