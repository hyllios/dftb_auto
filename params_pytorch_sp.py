import time
import torch

# [s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]

def get_hop_int(input_params=torch.zeros(7)):
    #  input_params[0]=V_sss,
    #  input_params[1]=V_sps,
    #  input_params[2]=V_pps,
    # input_params[3]=V_ppp,
    #  input_params[4]=l,
    #  input_params[5]=m
    #  input_params[6]=n
    #  input_params[7]=V_sss_2,
    #  input_params[8]=V_sps_2,
    #  input_params[9]=V_pps_2,
    # input_params[10]=V_ppp_2,
    # [[None for _ in range(10)] for __ in range(10)]
    input_params = input_params.permute(-1, 0, 1, 2)  # ???????????params_pytorch_sp.py
    hop_int_00 = input_params[0]
    hop_int_01 = input_params[4] * input_params[1]
    hop_int_02 = input_params[5] * input_params[1]
    hop_int_03 = input_params[6] * input_params[1]
    hop_int_10 = -input_params[4] * input_params[8]
    hop_int_20 = -input_params[5] * input_params[8]
    hop_int_30 = -input_params[6] * input_params[8]

    hop_int_11 = input_params[4] ** 2 * input_params[2] + (1. - input_params[4] ** 2) * input_params[3]
    hop_int_12 = input_params[4] * input_params[5] * (input_params[2] - input_params[3])
    hop_int_21 = hop_int_12
    hop_int_13 = input_params[4] * input_params[6] * (input_params[2] - input_params[3])
    hop_int_31 = hop_int_13
    hop_int_22 = input_params[5] ** 2 * input_params[2] + (1. - input_params[5] ** 2) * input_params[3]
    hop_int_23 = input_params[5] * input_params[6] * (input_params[2] - input_params[3])
    hop_int_32 = hop_int_23
    hop_int_33 = input_params[6] ** 2 * input_params[2] + (1. - input_params[6] ** 2) * input_params[3]
    return torch.stack((torch.stack((hop_int_00, hop_int_01, hop_int_02, hop_int_03)),
                        torch.stack((hop_int_10, hop_int_11, hop_int_12, hop_int_13)),
                        torch.stack((hop_int_20, hop_int_21, hop_int_22, hop_int_23)),
                        torch.stack((hop_int_30, hop_int_31, hop_int_32, hop_int_33))))



