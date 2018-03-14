import numpy as np
try:
    import constrained_mst as cmst
except ImportError:
    print("Constrained MST not found, can't use mst watersheds")


class DamWatershed(object):
    def __init__(self, offsets, stride,
                 seperating_channel=3, invert_dam_channels=True,
                 randomize_bounds=False):
        assert isinstance(offsets, list)
        # if we calculate stacked 2d superpixels from 3d affinity
        # maps, we must adjust the offsets by excludig all offsets
        # with z coordinates and make the rest 2d
        self.offsets = offsets
        self.dim = len(offsets[0])

        assert isinstance(stride, (int, list))
        if isinstance(stride, int):
            self.stride = self.dim * [stride]
        else:
            self.stride = stride
        assert len(stride) == self.dim

        assert seperating_channel < len(self.offsets)
        self.seperating_channel = seperating_channel
        self.invert_dam_channels = invert_dam_channels
        self.randomize_bounds = randomize_bounds

    def damws_superpixel(self, affinities):
        assert affinities.shape[0] >= len(self.offsets)
        # dam channels if ncessary
        if self.invert_dam_channels:
            affinities_ = affinities.copy()
            affinities_[self.seperating_channel:] *= -1
            affinities_[self.seperating_channel:] += 1
        else:
            affinities_ = affinities
        # sort all edges
        sorted_edges = np.argsort(affinities_.ravel())
        # run the mst watershed
        vol_shape = affinities_.shape[1:]
        mst = cmst.ConstrainedWatershed(np.array(vol_shape),
                                        self.offsets,
                                        self.seperating_channel,
                                        np.array(self.stride))
        mst.repulsive_ucc_mst_cut(sorted_edges, 0)
        if self.randomize_bounds:
            mst.compute_randomized_bounds()
        segmentation = mst.get_flat_label_image().reshape(vol_shape)
        max_label = segmentation.max()
        return segmentation, max_label

    def __call__(self, affinities):
        segmentation, _ = self.damws_superpixel(affinities)
        return segmentation
