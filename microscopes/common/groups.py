import numpy as np

class FixedNGroupManager(object):
    NOT_ASSIGNED = -1

    def __init__(self, n):
        self._n = n
        self._assignments = -1*np.ones(self._n, dtype=np.int)
        self._gid = 0
        self._gdata = {}

    def nentities(self):
        return self._n

    def ngroups(self):
        return len(self._gdata)

    def nentities_in_group(self, gid):
        return self._gdata[gid][0]

    def no_entities_assigned(self):
        return (self._assignments == -1).all()

    def all_entities_assigned(self):
        return not (self._assignments == -1).any()

    def empty_groups(self):
        for gid, (cnt, _) in self._gdata.iteritems():
            if not cnt:
                yield gid

    def group_data(self, gid):
        return self._gdata[gid][1]

    def groupiter(self):
        return self._gdata.iteritems()

    def create_group(self, gdata):
        """
        Creates the new group's groupid
        """
        gid = self._gid
        self._gid += 1
        self._gdata[gid] = [0, gdata]
        return gid

    def delete_group(self, gid):
        """
        Can only do this if the group is empty
        """
        assert not self.nentities_in_group(gid), 'group needs to be empty'
        del self._gdata[gid]

    def add_entity_to_group(self, gid, eid):
        """
        Add an unassigned entity to a group
        returns the group data
        """
        assert self._assignments[eid] == self.NOT_ASSIGNED
        self._assignments[eid] = gid
        ref = self._gdata[gid]
        ref[0] += 1
        return ref[1]

    def remove_entity_from_group(self, eid):
        """
        Remove the entity from its currently assigned group
        returns gid, gdata
        """
        assert self._assignments[eid] != self.NOT_ASSIGNED
        gid = self._assignments[eid]
        self._assignments[eid] = self.NOT_ASSIGNED
        ref = self._gdata[gid]
        ref[0] -= 1
        return gid, ref[1]
