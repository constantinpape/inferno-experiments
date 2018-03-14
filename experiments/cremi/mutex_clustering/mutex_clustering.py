import time
from concurrent import futures

import numpy as np
import vigra
# import sys
# sys.path.append('/home/papec/Work/software/bld/nifty/python')
import nifty.mws as nmws
import nifty.graph.rag as nrag
import z5py

# sys.path.append('/home/papec/Work/my_projects/cremi_tools')


def make_oversegmentation(pmaps, n_threads):
    hmap = np.mean(pmaps[1:3], axis=0) + np.mean(pmaps[4:6], axis=0)
    hmap /= 2

    seg = np.zeros_like(hmap, dtype='uint32')

    def run_ws_z(z):
        hmapz = np.clip(vigra.filters.gaussianSmoothing(hmap[z], sigma=2.), 0, 1)
        # hmapz -= hmapz.min()

        # seeds = vigra.analysis.localMaxima(hmap[z], allowPlateaus=True, allowAtBorder=True, marker=np.nan)
        # seeds = vigra.analysis.labelImageWithBackground(np.isnan(seeds).view('uint8'))

        # TODO looking at this, we are clearly doing something wrong .... this looks so perfect already !
        seeds = hmap[z] <= 0.05

        # seeds = hmapz == 0

        seeds = vigra.analysis.labelImageWithBackground(seeds.view('uint8'))
        seg_z, max_z = vigra.analysis.watershedsNew(hmapz, seeds=seeds)
        seg[z] = seg_z
        return max_z

    with futures.ThreadPoolExecutor(n_threads) as tp:
        tasks = [tp.submit(run_ws_z, z) for z in range(seg.shape[0])]
        offsets = np.array([t.result() for t in tasks], dtype='uint32')

    offsets = np.roll(offsets, 1)
    offsets[0] = 0
    offsets = np.cumsum(offsets)
    seg += offsets[:, None, None]

    return seg


def compute_features(seg, affs):
    import nifty.graph.rag as nrag
    offsets = [[-1, 0, 0], [0, -1, 0], [0, 0, -1],
               [-2, 0, 0], [0, -3, 0], [0, 0, -3],
               [-3, 0, 0], [0, -9, 0], [0, 0, -9],
               [-4, 0, 0], [0, -27, 0], [0, 0, -27]]
    rag = nrag.gridRag(seg, numberOfLabels=int(seg.max()) + 1)
    lr_uvs, local_features, lr_features = nrag.computeFeaturesAndNhFromAffinities(rag, affs, offsets)
    return rag, lr_uvs, local_features[:, 0], lr_features[:, 0]


def segment_sample(sample):

    aff_path = '%s' % sample

    print("Load affinities")
    affs = 1. - z5py.File(aff_path)['predictions/full_affs'][:]
    # affs = 1. - vigra.readHDF5('./sampleB+_affs_cut.h5', 'data')
    print("done")

    # TODO multi-threaded
    print("making oversegmentation")
    seg = make_oversegmentation(affs, 8)
    print("done")

    # for z in range(seg.shape[0]):
    #     print(seg[z].min(), seg[z].max(), seg[z].max() - seg[z].min())
    # quit()

    print("computing features")
    rag, lr_uvs, local_prob, lr_prob = compute_features(seg, affs)
    print("done")
    assert rag.numberOfEdges == len(local_prob)
    assert len(lr_uvs) == len(lr_prob)

    uvs = rag.uvIds()
    n_nodes = rag.numberOfNodes
    assert lr_uvs.max() + 1 == n_nodes

    print("compute mutex clustering")
    # TODO do I need to invert the lr weights ?!
    lr_prob = 1. - lr_prob
    t0 = time.time()
    node_labeling = nmws.computeMwsClustering(n_nodes,
                                              uvs.astype('uint32'), lr_uvs.astype('uint32'),
                                              local_prob, lr_prob)
    assert len(node_labeling) == n_nodes
    print("done in", time.time() - t0, "s")

    # get segmentation
    mws_seg = nrag.projectScalarNodeDataToPixels(rag, node_labeling)
    out_path = '' % sample
    vigra.writeHDF5(mws_seg, out_path, 'volumes/labels/neuron_ids', compression='gzip')


if __name__ == '__main__':
    segment_sample('A+')
