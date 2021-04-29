
from jax.numpy import sqrt
import jax.numpy as np
import numpy
from jax.api import jit
import time


#[s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]

@jit
def get_hop_int(input_params=np.zeros(14)):
    #  input_params[0]=V_sss,
    #  input_params[1]=V_sps,
    #  input_params[2]=V_pps,
    # input_params[3]=V_ppp,
    # input_params[4]=V_sds,
    #  input_params[5]=V_pds,
    #  input_params[6]=V_pdp,
    #  input_params[7]=V_dds,
    #  input_params[8]=V_ddp,
    #  input_params[9]=V_ddd,
    #  input_params[10]=l,
    #  input_params[11]=m
    #  input_params[12]=n)
    hop_int = np.array(numpy.zeros((9, 9)))  # [[None for _ in range(10)] for __ in range(10)]

    hop_int = hop_int.at[0, 0].set(input_params[0])
    hop_int = hop_int.at[0, 1].set(input_params[10] * input_params[1])
    hop_int = hop_int.at[0, 2].set(input_params[11] * input_params[1])
    hop_int = hop_int.at[0, 3].set(input_params[12] * input_params[1])
    hop_int = hop_int.at[1, 0].set(-hop_int[0][1])
    hop_int = hop_int.at[2, 0].set(-hop_int[0][2])
    hop_int = hop_int.at[3, 0].set(-hop_int[0][3])
    hop_int = hop_int.at[0, 4].set(sqrt(3) * input_params[10] * input_params[11] * input_params[4])
    hop_int = hop_int.at[0, 5].set(sqrt(3) * input_params[11] * input_params[12] * input_params[4])
    hop_int = hop_int.at[0, 6].set(sqrt(3) * input_params[10] * input_params[12] * input_params[4])
    hop_int = hop_int.at[4, 0].set(hop_int[0][4])
    hop_int = hop_int.at[5, 0].set(hop_int[0][5])
    hop_int = hop_int.at[6, 0].set(hop_int[0][6])
    hop_int = hop_int.at[0, 7].set(sqrt(3) / 2. * (input_params[10]**2 - input_params[11]**2) * input_params[4])
    hop_int = hop_int.at[7, 0].set(hop_int[0][7])
    hop_int = hop_int.at[0, 8].set((input_params[12]**2 - 0.5 * (input_params[10]**2 + input_params[11]**2)) * input_params[4])
    hop_int = hop_int.at[8, 0].set(hop_int[0][8])

    hop_int = hop_int.at[1, 1].set(input_params[10]**2 * input_params[2] + (1. - input_params[10]**2) * input_params[3])
    hop_int = hop_int.at[1, 2].set(input_params[10] * input_params[11] * (input_params[2] - input_params[3]))
    hop_int = hop_int.at[2, 1].set(hop_int[1][2])
    hop_int = hop_int.at[1, 3].set(input_params[10] * input_params[12] * (input_params[2] - input_params[3]))
    hop_int = hop_int.at[3, 1].set(hop_int[1][3])

    hop_int = hop_int.at[1, 4].set(sqrt(3) * input_params[10]**2 * input_params[11] * input_params[5] + input_params[11] * (1. - 2 * input_params[10]**2) * input_params[6])
    hop_int = hop_int.at[1, 5].set(input_params[10] * input_params[11] * input_params[12] * (sqrt(3) * input_params[5] - 2 * input_params[6]))
    hop_int = hop_int.at[1, 6].set(sqrt(3) * input_params[10]**2 * input_params[12] * input_params[5] + input_params[12] * (1. - 2 * input_params[10]**2) * input_params[6])
    hop_int = hop_int.at[4, 1].set(-hop_int[1][4])
    hop_int = hop_int.at[5, 1].set(-hop_int[1][5])
    hop_int = hop_int.at[6, 1].set(-hop_int[1][6])

    hop_int = hop_int.at[1, 7].set(0.5 * sqrt(3) * input_params[10] * (input_params[10]**2 - input_params[11]**2) * input_params[5] + input_params[10] * (
        1. - input_params[10]**2 + input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[1, 8].set(input_params[10] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] - sqrt(3) * input_params[10] * input_params[12]**2 * input_params[6])
    hop_int = hop_int.at[7, 1].set(-hop_int[1][7])
    hop_int = hop_int.at[8, 1].set(-hop_int[1][8])

    hop_int = hop_int.at[2, 2].set(input_params[11]**2 * input_params[2] + (1. - input_params[11]**2) * input_params[3])
    hop_int = hop_int.at[2, 3].set(input_params[11] * input_params[12] * (input_params[2] - input_params[3]))
    hop_int = hop_int.at[3, 2].set(hop_int[2][3])

    hop_int = hop_int.at[2, 4].set(sqrt(3) * input_params[11]**2 * input_params[10] * input_params[5] + input_params[10] * (1. - 2 * input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[2, 5].set(sqrt(3) * input_params[11]**2 * input_params[12] * input_params[5] + input_params[12] * (1. - 2 * input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[2, 6].set(input_params[10] * input_params[11] * input_params[12] * (sqrt(3) * input_params[5] - 2 * input_params[6]))
    hop_int = hop_int.at[4, 2].set(-hop_int[2][4])
    hop_int = hop_int.at[5, 2].set(-hop_int[2][5])
    hop_int = hop_int.at[6, 2].set(-hop_int[2][6])

    hop_int = hop_int.at[2, 7].set(0.5 * sqrt(3) * input_params[11] * (input_params[10]**2 - input_params[11]**2) * input_params[5] - input_params[11] * (
        1. + input_params[10]**2 - input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[2, 8].set(input_params[11] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] - sqrt(3) * input_params[11] * input_params[12]**2 * input_params[6])
    hop_int = hop_int.at[7, 2].set(-hop_int[2][7])
    hop_int = hop_int.at[8, 2].set(-hop_int[2][8])

    hop_int = hop_int.at[3, 3].set(input_params[12]**2 * input_params[2] + (1. - input_params[12]**2) * input_params[3])

    hop_int = hop_int.at[3, 4].set(input_params[10] * input_params[11] * input_params[12] * (sqrt(3) * input_params[5] - 2 * input_params[6]))
    hop_int = hop_int.at[3, 5].set(sqrt(3) * input_params[12]**2 * input_params[11] * input_params[5] + input_params[11] * (1. - 2 * input_params[12]**2) * input_params[6])
    hop_int = hop_int.at[3, 6].set(sqrt(3) * input_params[12]**2 * input_params[10] * input_params[5] + input_params[10] * (1. - 2 * input_params[12]**2) * input_params[6])
    hop_int = hop_int.at[4, 3].set(-hop_int[3][4])
    hop_int = hop_int.at[5, 3].set(-hop_int[3][5])
    hop_int = hop_int.at[6, 3].set(-hop_int[3][6])

    hop_int = hop_int.at[3, 7].set(0.5 * sqrt(3) * input_params[12] * (input_params[10]**2 - input_params[11]**2) * input_params[5] - input_params[12] * (
        input_params[10]**2 - input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[3, 8].set(input_params[12] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] + sqrt(3) * input_params[12] * (input_params[10]**2 +
                                                                 input_params[11]**2) * input_params[6])
    hop_int = hop_int.at[7, 3].set(-hop_int[3][7])
    hop_int = hop_int.at[8, 3].set(-hop_int[3][8])

    hop_int = hop_int.at[4, 4].set(input_params[10]**2 * input_params[11]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[10]**2 + input_params[11]**2) * input_params[8] + input_params[12]**2 * input_params[9])
    hop_int = hop_int.at[5, 5].set(input_params[11]**2 * input_params[12]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[11]**2 + input_params[12]**2) * input_params[8] + input_params[10]**2 * input_params[9])
    hop_int = hop_int.at[6, 6].set(input_params[12]**2 * input_params[10]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[12]**2 + input_params[10]**2) * input_params[8] + input_params[11]**2 * input_params[9])

    hop_int = hop_int.at[4, 5].set(input_params[10] * input_params[11]**2 * input_params[12] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[10] * input_params[12] * (input_params[8] - input_params[9]))
    hop_int = hop_int.at[4, 6].set(input_params[12] * input_params[10]**2 * input_params[11] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[12] * input_params[11] * (input_params[8] - input_params[9]))
    hop_int = hop_int.at[5, 6].set(input_params[11] * input_params[12]**2 * input_params[10] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[11] * input_params[10] * (input_params[8] - input_params[9]))
    hop_int = hop_int.at[5, 4].set(hop_int[4][5])
    hop_int = hop_int.at[6, 4].set(hop_int[4][6])
    hop_int = hop_int.at[6, 5].set(hop_int[5][6])

    hop_int = hop_int.at[4, 7].set(0.5 * input_params[10] * input_params[11] * (input_params[10]**2 - input_params[11]**2) * (3 * input_params[7] - 4 * input_params[8] +
                                                   input_params[9]))
    hop_int = hop_int.at[5, 7].set(0.5 * input_params[11] * input_params[12] * ((input_params[10]**2 - input_params[11]**2) *
                                   (3 * input_params[7] - 4 * input_params[8] + input_params[9]) - 2 *
                                   (input_params[8] - input_params[9])))
    hop_int = hop_int.at[6, 7].set(0.5 * input_params[12] * input_params[10] * ((input_params[10]**2 - input_params[11]**2) *
                                   (3 * input_params[7] - 4 * input_params[8] + input_params[9]) + 2 *
                                   (input_params[8] - input_params[9])))

    hop_int = hop_int.at[7, 4].set(hop_int[4][7])
    hop_int = hop_int.at[7, 5].set(hop_int[5][7])
    hop_int = hop_int.at[7, 6].set(hop_int[6][7])

    hop_int = hop_int.at[4, 8].set(sqrt(3) * (input_params[10] * input_params[11] * (input_params[12]**2 - 0.5 * (input_params[10]**2 + input_params[11]**2)) * input_params[7] -
                               2 * input_params[10] * input_params[11] * input_params[12]**2 * input_params[8] + 0.5 * input_params[10] * input_params[11] *
                               (1. + input_params[12]**2) * input_params[9]))
    hop_int = hop_int.at[5, 8].set(sqrt(3) * (input_params[11] * input_params[12] * (input_params[12]**2 - 0.5 *
                                        (input_params[10]**2 + input_params[11]**2)) * input_params[7] + input_params[11] * input_params[12] *
                               (input_params[10]**2 + input_params[11]**2 - input_params[12]**2) * input_params[8] - 0.5 * input_params[11] * input_params[12] *
                               (input_params[10]**2 + input_params[11]**2) * input_params[9]))
    hop_int = hop_int.at[6, 8].set(sqrt(3) * (input_params[12] * input_params[10] * (input_params[12]**2 - 0.5 *
                                        (input_params[10]**2 + input_params[11]**2)) * input_params[7] + input_params[12] * input_params[10] *
                               (input_params[10]**2 + input_params[11]**2 - input_params[12]**2) * input_params[8] - 0.5 * input_params[12] * input_params[10] *
                               (input_params[10]**2 + input_params[11]**2) * input_params[9]))
    hop_int = hop_int.at[8, 4].set(hop_int[4][8])
    hop_int = hop_int.at[8, 5].set(hop_int[5][8])
    hop_int = hop_int.at[8, 6].set(hop_int[6][8])

    hop_int = hop_int.at[7, 7].set(0.25 * (input_params[10]**2 - input_params[11]**2)**2 * (
        3 * input_params[7] - 4 * input_params[8] + input_params[9]) + (input_params[10]**2 + input_params[11]**2) * input_params[8] + input_params[12]**2 * input_params[9])
    hop_int = hop_int.at[8, 8].set(0.75 * (input_params[10]**2 + input_params[11]**2)**2 * input_params[9] + 3 * (
        input_params[10]**2 + input_params[11]**2) * input_params[12]**2 * input_params[8] + 0.25 * (input_params[10]**2 + input_params[11]**2 - 2 * input_params[12]**2)**2 * input_params[7])
    hop_int = hop_int.at[7, 8].set(sqrt(3) * 0.25 * (input_params[10]**2 - input_params[11]**2) * (
        input_params[12]**2 * (2 * input_params[7] - 4 * input_params[8] + input_params[9]) + input_params[9] - (input_params[10]**2 + input_params[11]**2) * input_params[7]))
    hop_int = hop_int.at[8, 7].set(sqrt(3) * 0.25 * (input_params[10]**2 - input_params[11]**2) * (
        input_params[12]**2 * (2 * input_params[7] - 4 * input_params[8] + input_params[9]) + input_params[9] - (input_params[10]**2 + input_params[11]**2) * input_params[7]))
    return hop_int
    
    

