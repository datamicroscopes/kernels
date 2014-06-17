import numpy as np

from distributions.lp.clustering import PitmanYor
from distributions.lp.mixture import MixtureIdTracker

class DPMM(object):
    EMPTY_GROUP_COUNT = 1 # hacky

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
            return DPMM.FeatureGroup(g, self._shared)
        def __str__(self):
            return str(self._shared)
        def __repr__(self):
            return repr(self._shared)

    def __init__(self):
        self._clustering = None
        self._idtracker = None
        self._featuretypes = None
        self._features = None
        self._groups = None

    def init(self, clusterhp, featuretypes, featurehps):
        self._clustering = PitmanYor.Mixture()
        self._clustering_shared = PitmanYor.from_dict(clusterhp)
        self._clustering.init(self._clustering_shared, [0]*self.EMPTY_GROUP_COUNT)
        self._idtracker = MixtureIdTracker()
        self._idtracker.init(self.EMPTY_GROUP_COUNT)
        self._featuretypes = featuretypes
        def init_and_load_feature(arg):
            typ, hp = arg
            shared = typ.Shared()
            shared.load(hp)
            return DPMM.Feature(typ, shared)
        self._features = map(init_and_load_feature, zip(featuretypes, featurehps))
        self._groups = {}

    def add_value(self, groupid, y):
        packed_groupid = self._idtracker.global_to_packed(groupid)
        assert packed_groupid < len(self._clustering), 'invalid groupid'
        def _add_value_existing(group, y):
            for g, yi in zip(group, y):
                g.add_value(yi)
        group_added = self._clustering.add_value(self._clustering_shared, packed_groupid)
        if group_added:
            assert groupid not in self._groups
            group = [f.new_group() for f in self._features]
            _add_value_existing(group, y)
            self._groups[groupid] = group
            self._idtracker.add_group()
        else:
            _add_value_existing(self._groups[groupid], y)

    def remove_value(self, groupid, y):
        packed_groupid = self._idtracker.global_to_packed(groupid)
        assert packed_groupid < len(self._clustering), 'invalid groupid'
        group_removed = self._clustering.remove_value(self._clustering_shared, packed_groupid)
        if group_removed:
            assert groupid in self._groups
            del self._groups[groupid]
            self._idtracker.remove_group(packed_groupid)
        else:
            for g, yi in zip(self._groups[groupid], y):
                g.remove_value(yi)

    def score_value(self, y):
        scores = np.zeros(len(self._clustering), dtype=np.float32)
        self._clustering.score_value(self._clustering_shared, scores)
        for groupid, group in self._groups.iteritems():
            scores[self._idtracker.global_to_packed(groupid)] += sum(g.score_value(yi) for g, yi in zip(group, y))
        return [self._idtracker.packed_to_global(i) for i in xrange(len(self._clustering))], scores
