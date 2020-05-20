import numpy as np
import matplotlib.pyplot as plt

import test_utils

N = 5
num_samples = 10000

NADE_prob = np.loadtxt("NADE_probability")
NADE_samples = np.loadtxt("NADE_samples", dtype=int)

prob_inds, prob_samples = test_utils.gen_samples(num_samples, N, NADE_prob)
sample_inds = test_utils.gen_inds_from_samples(NADE_samples)

prob_uniques, prob_counts = np.unique(prob_inds, return_counts=True)
sample_uniques, sample_counts = np.unique(sample_inds, return_counts=True)

prob_counts = prob_counts / len(prob_inds)
sample_counts = sample_counts / len(sample_inds)


plt.figure()
plt.bar(prob_uniques+0.1, prob_counts, color='blue',
        label="NADE_probability samples", align='center', width=0.25)
plt.bar(sample_uniques-0.1, sample_counts, color='green',
        label="NADE_samples samples", align='center', width=0.25)
plt.xlabel("Basis state index")
plt.ylabel("Fractional frequency")
plt.legend()
plt.xticks(np.arange(0,2**N,2))
plt.savefig("test_samples.pdf", dpi=500, bbox_inches='tight')

