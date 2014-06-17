import numpy as np

from microscopes.common.groups import FixedNGroupManager
from distributions.dbg.random import sample_discrete_log

####
# Some utilities for dealing with distributions which bind the shared values so
# we don't have to keep passing them around
####

class FeatureGroup(object):
    def __init__(self, group, shared):
        self._group = group
        self._shared = shared
    def add_value(self, value):
        self._group.add_value(self._shared, value)
    def remove_value(self, value):
        self._group.remove_value(self._shared, value)
    def score_value(self, value):
        return self._group.score_value(self._shared, value)
    def __str__(self):
        return str(self._group)
    def __repr__(self):
        return repr(self._group)

class Feature(object):
    def __init__(self, typ, shared):
        self._typ = typ
        self._shared = shared
    def new_group(self):
        g = self._typ.Group()
        g.init(self._shared)
        return FeatureGroup(g, self._shared)
    def __str__(self):
        return str(self._shared)
    def __repr__(self):
        return repr(self._shared)

class DPMM(object):

    def __init__(self, n, clusterhp, featuretypes, featurehps):
        self._groups = FixedNGroupManager(n)
        self._alpha = clusterhp['alpha'] # CRP alpha
        self._featuretypes = featuretypes
        def init_and_load_feature(arg):
            typ, hp = arg
            shared = typ.Shared()
            shared.load(hp)
            return Feature(typ, shared)
        self._features = map(init_and_load_feature, zip(featuretypes, featurehps))

    def empty_groups(self):
        return self._groups.empty_groups()

    def ngroups(self):
        return self._groups.ngroups()

    def nentities(self):
        return self._groups.nentities()

    def nentities_in_group(self, gid):
        return self._groups.nentities_in_group(gid)

    def is_group_empty(self, gid):
        return not self._groups.nentities_in_group(gid)

    def create_group(self):
        """
        returns gid
        """
        gdata = tuple(f.new_group() for f in self._features)
        return self._groups.create_group(gdata)

    def delete_group(self, gid):
        self._groups.delete_group(gid)

    def add_entity_to_group(self, gid, eid, y):
        gdata = self._groups.add_entity_to_group(gid, eid)
        for g, yi in zip(gdata, y):
            g.add_value(yi)

    def remove_entity_from_group(self, eid, y):
        """
        returns gid
        """
        gid, gdata = self._groups.remove_entity_from_group(eid)
        for g, yi in zip(gdata, y):
            g.remove_value(yi)
        return gid

    def score_value(self, y):
        """
        returns idmap, scores
        """
        scores = np.zeros(self._groups.ngroups(), dtype=np.float)
        idmap = [0]*self._groups.ngroups()
        n = self._groups.nentities()
        for idx, (gid, (cnt, gdata)) in enumerate(self._groups.groupiter()):
            lg_term1 = np.log((self._alpha if not cnt else cnt)/(n-1-self._alpha)) # CRP
            lg_term2 = sum(g.score_value(yi) for g, yi in zip(gdata, y))
            scores[idx] = lg_term1 + lg_term2
            idmap[idx] = gid
        return idmap, scores

    def bootstrap(self, it):
        """
        bootstraps assignments
        """
        assert not self.ngroups()
        assert self._groups.no_entities_assigned()

        ei0, y0 = next(it)
        gid0 = self.create_group()
        self.add_entity_to_group(gid0, ei0, y0)
        empty_gid = self.create_group()
        for ei, yi in it:
            idmap, scores = self.score_value(yi)
            gid = idmap[sample_discrete_log(scores)]
            self.add_entity_to_group(gid, ei, yi)
            if gid == empty_gid:
                empty_gid = self.create_group()

        assert self._groups.all_entities_assigned()
