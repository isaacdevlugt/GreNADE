import numpy as np
from qucumber.utils import unitaries
from itertools import product

# Some code directly from https://github.com/emerali/rand_wvfn_sampler/blob/master/data_gen_py.ipynb

'''
Check sampling algorithm by:

- generating samples from DMRG wavefunction
- take the wavefunction from DMRG and directly sample it
- take the ED wavefunction and directly sample it
- bin everything and compare

Do this for every basis
'''


def generate_hilbert_space(size):
    dim = np.arange(2 ** size)
    space = (((dim[:, None] & (1 << np.arange(size)))) > 0)[:, ::-1]
    space = space.astype(int)
    return space


def get_samples_from_psi_indices(indices, N):
    return (((indices[:, None] & (1 << np.arange(N)))) > 0)[:, ::-1].astype(int)


def gen_samples(num_samples, N, probs):
    indices = np.random.choice(len(probs), size=num_samples, p=probs)
    return indices, get_samples_from_psi_indices(indices, N)


def gen_inds_from_samples(samples):
    inds = np.zeros(len(samples))
    for i in range(len(samples)):
        inds[i] = int("".join(str(i) for i in samples[i]), base=2)
    return inds.astype(int)


def convert_torch_cplx(tensor):
    real_part = tensor[0].detach().numpy()
    imag_part = tensor[1].detach().numpy()

    return real_part + (1j * imag_part)


def gen_all_bases(unitary_dict, num_sites):
    local_bases = unitary_dict.keys()
    return list("".join(i) for i in product(local_bases, repeat=num_sites))


def rotate_psi(unitary_dict, basis, psi):
    U1 = unitary_dict[basis[0]]
    U2 = unitary_dict[basis[1]]
    unitary = np.kron(U1, U2)
    return np.dot(unitary, psi)


def gen_data(N, num_samples_per_basis, unitary_dict, DMRG_psi, ED_psi):

    size = 2 ** N
    vis = generate_hilbert_space(N)

    all_bases = gen_all_bases(unitary_dict, N)

    tr_bases = np.zeros((len(all_bases), num_sites), dtype=str)
    samples = np.zeros(
        (len(all_bases), num_samples_per_basis, num_sites), dtype=int)

    for i, basis in enumerate(tqdm(all_bases)):
        tr_bases[i, :] = np.array(list(basis))
        samples[i, :, :] = gen_samples(
            num_samples_per_basis, num_sites, psi, basis, unitary_dict, vis)

    tr_bases = np.repeat(
        tr_bases[:, None, :], num_samples_per_basis, axis=1).reshape(-1, num_sites)
    samples = samples.reshape(-1, num_sites)

    all_bases = np.array(list(map(list, all_bases)))

    return all_bases, tr_bases, samples, psi
