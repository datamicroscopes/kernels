import numpy as np

def mh_hp(m, hpdfs, hcondpdfs, hsamplers):
    """
    run one iteration of metropolis hastings for hyperparameter estimation
    """
    # XXX: this can be done in parallel
    for fi, (hpdf, hcondpdf, hsamp) in enumerate(zip(hpdfs, hcondpdfs, hsamplers)):
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
