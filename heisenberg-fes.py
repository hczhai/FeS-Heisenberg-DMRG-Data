# -*- coding: utf-8 -*-
"""DMRG of S=1/2 and S=5/2 Heisenberg Models
"""

# !pip install https://github.com/block-hczhai/block2-preview/releases/download/p0.5.1rc9/block2-0.5.1rc9-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl

import numpy as np
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

def perform_dmrg(local_spin, global_spins, topology, n_sites):

    driver = DMRGDriver(scratch='./tmp', symm_type=SymmetryTypes.SU2,
                        n_threads=4)
    driver.initialize_system(n_sites=n_sites, spin=0, heis_twos=local_spin)

    b = driver.expr_builder()
    for (i, j, v) in topology:
        assert i != j
        b.add_term("(T+T)0", [i, j], -np.sqrt(3) / 2 * v)
    heis_mpo = driver.get_mpo(b.finalize(), iprint=0)

    bond_dims = [200] * 5 + [400] * 5 + [800] * 5
    noises = [1E-4] * 5 + [1E-5] * 5 + [0]
    dav_thrds = [1E-7] * 5 + [1E-9] * 5 + [1E-10]

    all_eners = []

    print("\n === S = %d/2 Heisenberg %d-Site Model === \n" %
          (local_spin, n_sites))
    for spin in global_spins:
        target = driver.target
        target.twos = target.twos_low = spin
        ket = driver.get_random_mps(tag='GS', bond_dim=200, nroots=12)
        eners = driver.dmrg(heis_mpo, ket, n_sweeps=20, iprint=0,
                                  bond_dims=bond_dims, noises=noises,
                                  thrds=dav_thrds, dav_max_iter=100, cutoff=0)
        all_eners.append(eners)
        print('2S = %d : E =' % spin, ('%14.8f' * len(eners)) % tuple(eners))

    print("\nRelative energies:\n")
    ener_min = np.min([np.min(x) for x in all_eners])
    for spin, eners in zip(global_spins, all_eners):
        eners = np.array(eners) - ener_min
        print('2S = %d : E =' % spin, ('%14.8f' * len(eners)) % tuple(eners))

n_sites = 4
topology = [
    (1, 2, 1.17),
    (3, 4, 1.17),
    (1, 4, 1.00),
    (2, 3, 1.00),
    (1, 3, 1.00),
    (2, 4, 1.00),
]
topology = [(i - 1, j - 1, k) for i, j, k in topology]

perform_dmrg(local_spin=5, global_spins=[0, 2, 4],
             topology=topology, n_sites=n_sites)

perform_dmrg(local_spin=1, global_spins=[0, 2, 4],
             topology=topology, n_sites=n_sites)

n_sites = 8
topology = [
    (1, 2, 1.00),
    (2, 3, 1.00),
    (3, 4, 1.00),
    (1, 4, 1.00),
    (1, 3, 1.00),
    (2, 4, 1.00),

    (5, 6, 1.00),
    (5, 7, 1.00),
    (6, 8, 1.00),
    (7, 8, 1.00),
    (5, 8, 1.00),
    (6, 7, 1.00),

    (2, 5, 1.00),
    (2, 6, 1.00),
    (2, 7, 1.00),

    (3, 5, 1.00),
    (3, 6, 1.00),
    (3, 7, 1.00),

    (4, 5, 1.00),
    (4, 6, 1.00),
    (4, 7, 1.00),
]
topology = [(i - 1, j - 1, k) for i, j, k in topology]

perform_dmrg(local_spin=5, global_spins=[0, 2, 4, 6, 8],
             topology=topology, n_sites=n_sites)

perform_dmrg(local_spin=1, global_spins=[0, 2, 4, 6, 8],
             topology=topology, n_sites=n_sites)

