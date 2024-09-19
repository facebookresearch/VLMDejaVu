"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import numpy as np
import faiss
import time
import pickle


class FAISSIndex():
    """
    FAISS index class to build multi-GPU index and search.
    """
    def __init__(self, qrs=100, dims=512, bs=512, tot=10000, batch_mode=False):
        #faiss.cvar.distance_compute_blas_threshold = 40
        self.batch_mode = batch_mode
        self.qrs = qrs
        self.dims = dims
        self.bs = bs
        self.tot = tot
        self.save_file = 'tmp_index.p'
        self.p = faiss.GpuMultipleClonerOptions()
        self.p.shard = True

    def build_index(self):
        arr = np.random.rand(self.bs * self.tot, self.dims)
        V = np.random.randint(self.bs * self.tot, size=(self.qrs,))
        V_proj = np.array(arr[V], dtype='float32')
        index = faiss.IndexFlatL2(self.dims)
        index = faiss.index_cpu_to_all_gpus(index, self.p)
        if self.batch_mode:
            t0 = time.time()
            for batch in np.split(arr, self.tot):
                index.add(np.array(batch, dtype='float32'))
            t1 = time.time()
            print(f"Batch indexing took {t1-t0} secs.")
        else:
            t0 = time.time()
            index.add(np.array(arr, dtype='float32'))
            t1 = time.time()
            print(f"Full indexing took {t1-t0} secs.")
        pickle.dump(
            [faiss.index_gpu_to_cpu(index), V, V_proj],
            open(self.save_file, 'wb')
        )

    def search(self):
        index, V, V_proj = pickle.load(open(self.save_file, 'rb'))
        index = faiss.index_cpu_to_all_gpus(index, self.p)
        t0 = time.time()
        D, I = index.search(V_proj, 1)
        ct = 0
        t1 = time.time()
        for i in range(self.qrs):
            if I[i][0] == V[i]:
                ct += 1
        print(f"{ct} out of {self.qrs} matches")
        print(f"Index search took {t1-t0} secs.")


if __name__ == '__main__':
    # GPU full index takes 16.6 mins to index 50M+ records
    findex = FAISSIndex(
        qrs=100,
        dims=512,
        bs=512,
        tot=100000,
        batch_mode=False
    )
    findex.build_index()
    findex.search()
