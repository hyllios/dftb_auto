import time
import torch
from torch import sqrt as sqrt
# [s, px, py, pz, dxy, dyz, dxz, dx2-y2, dz2, S]

def get_hop_int(input_params=torch.zeros(13)):
    #  input_params[0]=V_sss,
    #  input_params[1]=V_sps,  !!!!!!  --> -1
    #  input_params[2]=V_pps,
    #  input_params[3]=V_ppp,
    #  input_params[4]=V_sds,  !!!!!!  --> +1
    #  input_params[5]=V_pds,  !!!!!!  --> -1
    #  input_params[6]=V_pdp,  !!!!!!  --> -1
    #  input_params[7]=V_dds,
    #  input_params[8]=V_ddp,
    #  input_params[9]=V_ddd,
    #  input_params[10]=l,
    #  input_params[11]=m
    #  input_params[12]=n
    #  input_params[13]=V_sss_2,
    #  input_params[14]=V_sps_2,  !!!!!!
    #  input_params[15]=V_pps_2,
    #  input_params[16]=V_ppp_2,
    #  input_params[17]=V_sds_2,  !!!!!!
    #  input_params[18]=V_pds_2,  !!!!!!
    #  input_params[19]=V_pdp_2,  !!!!!!
    #  input_params[20]=V_dds_2,
    #  input_params[21]=V_ddp_2,
    #  input_params[22]=V_ddd_2)

    input_params = input_params.permute(-1, 0, 1, 2)  # ???????????params_pytorch_sp.py

    hop_int_00 = (input_params[0])
    hop_int_01 = (input_params[10] * input_params[1])
    hop_int_02 = (input_params[11] * input_params[1])
    hop_int_03 = (input_params[12] * input_params[1])
    hop_int_10 = -(input_params[10] * input_params[14])
    hop_int_20 = -(input_params[11] * input_params[14])
    hop_int_30 = -(input_params[12] * input_params[14])

    hop_int_04 = (sqrt(torch.tensor(3.0)) * input_params[10] * input_params[11] * input_params[4])
    hop_int_05 = (sqrt(torch.tensor(3.0)) * input_params[11] * input_params[12] * input_params[4])
    hop_int_06 = (sqrt(torch.tensor(3.0)) * input_params[10] * input_params[12] * input_params[4])
    hop_int_40 = (sqrt(torch.tensor(3.0)) * input_params[10] * input_params[11] * input_params[17])
    hop_int_50 = (sqrt(torch.tensor(3.0)) * input_params[11] * input_params[12] * input_params[17])
    hop_int_60 = (sqrt(torch.tensor(3.0)) * input_params[10] * input_params[12] * input_params[17])

    hop_int_07 = (sqrt(torch.tensor(3.0)) / 2. * (input_params[10]**2 - input_params[11]**2) * input_params[4])
    hop_int_70 = (sqrt(torch.tensor(3.0)) / 2. * (input_params[10]**2 - input_params[11]**2) * input_params[17])
    hop_int_08 = ((input_params[12]**2 - 0.5 * (input_params[10]**2 + input_params[11]**2)) * input_params[4])
    hop_int_80 = ((input_params[12]**2 - 0.5 * (input_params[10]**2 + input_params[11]**2)) * input_params[17])

    hop_int_11 = (input_params[10]**2 * input_params[2] + (1. - input_params[10]**2) * input_params[3])
    hop_int_12 = (input_params[10] * input_params[11] * (input_params[2] - input_params[3]))
    hop_int_21 = (hop_int_12)
    hop_int_13 = (input_params[10] * input_params[12] * (input_params[2] - input_params[3]))
    hop_int_31 = (hop_int_13)

    hop_int_14 = (sqrt(torch.tensor(3.0)) * input_params[10]**2 * input_params[11] * input_params[5] + input_params[11] * (1. - 2 * input_params[10]**2) * input_params[6])
    hop_int_15 = (input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[5] - 2 * input_params[6]))
    hop_int_16 = (sqrt(torch.tensor(3.0)) * input_params[10]**2 * input_params[12] * input_params[5] + input_params[12] * (1. - 2 * input_params[10]**2) * input_params[6])
    hop_int_41 = -(sqrt(torch.tensor(3.0)) * input_params[10]**2 * input_params[11] * input_params[18] + input_params[11] * (1. - 2 * input_params[10]**2) * input_params[19])
    hop_int_51 = -(input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[18] - 2 * input_params[19]))
    hop_int_61 = -(sqrt(torch.tensor(3.0)) * input_params[10]**2 * input_params[12] * input_params[18] + input_params[12] * (1. - 2 * input_params[10]**2) * input_params[19])

    hop_int_17 = (0.5 * sqrt(torch.tensor(3.0)) * input_params[10] * (input_params[10]**2 - input_params[11]**2) * input_params[5] + input_params[10] * (
        1. - input_params[10]**2 + input_params[11]**2) * input_params[6])
    hop_int_18 = (input_params[10] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] - sqrt(torch.tensor(3.0)) * input_params[10] * input_params[12]**2 * input_params[6])
    hop_int_71 = -(0.5 * sqrt(torch.tensor(3.0)) * input_params[10] * (input_params[10]**2 - input_params[11]**2) * input_params[18] + input_params[10] * (
        1. - input_params[10]**2 + input_params[11]**2) * input_params[19])
    hop_int_81 = -(input_params[10] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[18] - sqrt(torch.tensor(3.0)) * input_params[10] * input_params[12]**2 * input_params[19])

    hop_int_22 = (input_params[11]**2 * input_params[2] + (1. - input_params[11]**2) * input_params[3])
    hop_int_23 = (input_params[11] * input_params[12] * (input_params[2] - input_params[3]))
    hop_int_32 = (hop_int_23)

    hop_int_24 = (sqrt(torch.tensor(3.0)) * input_params[11]**2 * input_params[10] * input_params[5] + input_params[10] * (1. - 2 * input_params[11]**2) * input_params[6])
    hop_int_25 = (sqrt(torch.tensor(3.0)) * input_params[11]**2 * input_params[12] * input_params[5] + input_params[12] * (1. - 2 * input_params[11]**2) * input_params[6])
    hop_int_26 = (input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[5] - 2 * input_params[6]))
    hop_int_42 = -(sqrt(torch.tensor(3.0)) * input_params[11]**2 * input_params[10] * input_params[18] + input_params[10] * (1. - 2 * input_params[11]**2) * input_params[19])
    hop_int_52 = -(sqrt(torch.tensor(3.0)) * input_params[11]**2 * input_params[12] * input_params[18] + input_params[12] * (1. - 2 * input_params[11]**2) * input_params[19])
    hop_int_62 = -(input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[18] - 2 * input_params[19]))

    hop_int_27 = (0.5 * sqrt(torch.tensor(3.0)) * input_params[11] * (input_params[10]**2 - input_params[11]**2) * input_params[5] - input_params[11] * (
        1. + input_params[10]**2 - input_params[11]**2) * input_params[6])
    hop_int_28 = (input_params[11] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] - sqrt(torch.tensor(3.0)) * input_params[11] * input_params[12]**2 * input_params[6])
    hop_int_72 = -(0.5 * sqrt(torch.tensor(3.0)) * input_params[11] * (input_params[10]**2 - input_params[11]**2) * input_params[18] - input_params[11] * (
        1. + input_params[10]**2 - input_params[11]**2) * input_params[19])
    hop_int_82 = -(input_params[11] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[18] - sqrt(torch.tensor(3.0)) * input_params[11] * input_params[12]**2 * input_params[19])

    hop_int_33 = (input_params[12]**2 * input_params[2] + (1. - input_params[12]**2) * input_params[3])

    hop_int_34 = (input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[5] - 2 * input_params[6]))
    hop_int_35 = (sqrt(torch.tensor(3.0)) * input_params[12]**2 * input_params[11] * input_params[5] + input_params[11] * (1. - 2 * input_params[12]**2) * input_params[6])
    hop_int_36 = (sqrt(torch.tensor(3.0)) * input_params[12]**2 * input_params[10] * input_params[5] + input_params[10] * (1. - 2 * input_params[12]**2) * input_params[6])
    hop_int_43 = -(input_params[10] * input_params[11] * input_params[12] * (sqrt(torch.tensor(3.0)) * input_params[18] - 2 * input_params[19]))
    hop_int_53 = -(sqrt(torch.tensor(3.0)) * input_params[12]**2 * input_params[11] * input_params[18] + input_params[11] * (1. - 2 * input_params[12]**2) * input_params[19])
    hop_int_63 = -(sqrt(torch.tensor(3.0)) * input_params[12]**2 * input_params[10] * input_params[18] + input_params[10] * (1. - 2 * input_params[12]**2) * input_params[19])

    hop_int_37 = (0.5 * sqrt(torch.tensor(3.0)) * input_params[12] * (input_params[10]**2 - input_params[11]**2) * input_params[5] - input_params[12] * (
        input_params[10]**2 - input_params[11]**2) * input_params[6])
    hop_int_38 = (input_params[12] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[5] + sqrt(torch.tensor(3.0)) * input_params[12] * (input_params[10]**2 +
                                                                 input_params[11]**2) * input_params[6])
    hop_int_73 = -(0.5 * sqrt(torch.tensor(3.0)) * input_params[12] * (input_params[10]**2 - input_params[11]**2) * input_params[18] - input_params[12] * (
        input_params[10]**2 - input_params[11]**2) * input_params[19])
    hop_int_83 = -(input_params[12] * (input_params[12]**2 - 0.5 *
                         (input_params[10]**2 + input_params[11]**2)) * input_params[18] + sqrt(torch.tensor(3.0)) * input_params[12] * (input_params[10]**2 +
                                                                 input_params[11]**2) * input_params[19])

    hop_int_44 = (input_params[10]**2 * input_params[11]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[10]**2 + input_params[11]**2) * input_params[8] + input_params[12]**2 * input_params[9])
    hop_int_55 = (input_params[11]**2 * input_params[12]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[11]**2 + input_params[12]**2) * input_params[8] + input_params[10]**2 * input_params[9])
    hop_int_66 = (input_params[12]**2 * input_params[10]**2 * (3 * input_params[7] - 4 * input_params[8] +
                                   input_params[9]) + (input_params[12]**2 + input_params[10]**2) * input_params[8] + input_params[11]**2 * input_params[9])

    hop_int_45 = (input_params[10] * input_params[11]**2 * input_params[12] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[10] * input_params[12] * (input_params[8] - input_params[9]))
    hop_int_46 = (input_params[12] * input_params[10]**2 * input_params[11] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[12] * input_params[11] * (input_params[8] - input_params[9]))
    hop_int_56 = (input_params[11] * input_params[12]**2 * input_params[10] * (3 * input_params[7] - 4 * input_params[8] +
                                    input_params[9]) + input_params[11] * input_params[10] * (input_params[8] - input_params[9]))
    hop_int_54 = (hop_int_45)
    hop_int_64 = (hop_int_46)
    hop_int_65 = (hop_int_56)

    hop_int_47 = (0.5 * input_params[10] * input_params[11] * (input_params[10]**2 - input_params[11]**2) * (3 * input_params[7] - 4 * input_params[8] +
                                                   input_params[9]))
    hop_int_57 = (0.5 * input_params[11] * input_params[12] * ((input_params[10]**2 - input_params[11]**2) *
                                   (3 * input_params[7] - 4 * input_params[8] + input_params[9]) - 2 *
                                   (input_params[8] - input_params[9])))
    hop_int_67 = (0.5 * input_params[12] * input_params[10] * ((input_params[10]**2 - input_params[11]**2) *
                                   (3 * input_params[7] - 4 * input_params[8] + input_params[9]) + 2 *
                                   (input_params[8] - input_params[9])))

    hop_int_74 = (hop_int_47)
    hop_int_75 = (hop_int_57)
    hop_int_76 = (hop_int_67)

    hop_int_48 = (sqrt(torch.tensor(3.0)) * (input_params[10] * input_params[11] * (input_params[12]**2 - 0.5 * (input_params[10]**2 + input_params[11]**2)) * input_params[7] -
                               2 * input_params[10] * input_params[11] * input_params[12]**2 * input_params[8] + 0.5 * input_params[10] * input_params[11] *
                               (1. + input_params[12]**2) * input_params[9]))
    hop_int_58 = (sqrt(torch.tensor(3.0)) * (input_params[11] * input_params[12] * (input_params[12]**2 - 0.5 *
                                        (input_params[10]**2 + input_params[11]**2)) * input_params[7] + input_params[11] * input_params[12] *
                               (input_params[10]**2 + input_params[11]**2 - input_params[12]**2) * input_params[8] - 0.5 * input_params[11] * input_params[12] *
                               (input_params[10]**2 + input_params[11]**2) * input_params[9]))
    hop_int_68 = (sqrt(torch.tensor(3.0)) * (input_params[12] * input_params[10] * (input_params[12]**2 - 0.5 *
                                        (input_params[10]**2 + input_params[11]**2)) * input_params[7] + input_params[12] * input_params[10] *
                               (input_params[10]**2 + input_params[11]**2 - input_params[12]**2) * input_params[8] - 0.5 * input_params[12] * input_params[10] *
                               (input_params[10]**2 + input_params[11]**2) * input_params[9]))
    hop_int_84 = (hop_int_48)
    hop_int_85 = (hop_int_58)
    hop_int_86 = (hop_int_68)

    hop_int_77 = (0.25 * (input_params[10]**2 - input_params[11]**2)**2 * (
        3 * input_params[7] - 4 * input_params[8] + input_params[9]) + (input_params[10]**2 + input_params[11]**2) * input_params[8] + input_params[12]**2 * input_params[9])
    hop_int_88 = (0.75 * (input_params[10]**2 + input_params[11]**2)**2 * input_params[9] + 3 * (
        input_params[10]**2 + input_params[11]**2) * input_params[12]**2 * input_params[8] + 0.25 * (input_params[10]**2 + input_params[11]**2 - 2 * input_params[12]**2)**2 * input_params[7])
    hop_int_78 = (sqrt(torch.tensor(3.0)) * 0.25 * (input_params[10]**2 - input_params[11]**2) * (
        input_params[12]**2 * (2 * input_params[7] - 4 * input_params[8] + input_params[9]) + input_params[9] - (input_params[10]**2 + input_params[11]**2) * input_params[7]))
    hop_int_87 = (sqrt(torch.tensor(3.0)) * 0.25 * (input_params[10]**2 - input_params[11]**2) * (
        input_params[12]**2 * (2 * input_params[7] - 4 * input_params[8] + input_params[9]) + input_params[9] - (input_params[10]**2 + input_params[11]**2) * input_params[7]))
    return torch.stack((torch.stack((hop_int_00, hop_int_01, hop_int_02, hop_int_03, hop_int_04, hop_int_05, hop_int_06, hop_int_07, hop_int_08)),
                        torch.stack((hop_int_10, hop_int_11, hop_int_12, hop_int_13, hop_int_14, hop_int_15, hop_int_16, hop_int_17, hop_int_18)),
                        torch.stack((hop_int_20, hop_int_21, hop_int_22, hop_int_23, hop_int_24, hop_int_25, hop_int_26, hop_int_27, hop_int_28)),
                        torch.stack((hop_int_30, hop_int_31, hop_int_32, hop_int_33, hop_int_34, hop_int_35, hop_int_36, hop_int_37, hop_int_38)),
                        torch.stack((hop_int_40, hop_int_41, hop_int_42, hop_int_43, hop_int_44, hop_int_45, hop_int_46, hop_int_47, hop_int_48)),
                        torch.stack((hop_int_50, hop_int_51, hop_int_52, hop_int_53, hop_int_54, hop_int_55, hop_int_56, hop_int_57, hop_int_58)),
                        torch.stack((hop_int_60, hop_int_61, hop_int_62, hop_int_63, hop_int_64, hop_int_65, hop_int_66, hop_int_67, hop_int_68)),
                        torch.stack((hop_int_70, hop_int_71, hop_int_72, hop_int_73, hop_int_74, hop_int_75, hop_int_76, hop_int_77, hop_int_78)),
                        torch.stack((hop_int_80, hop_int_81, hop_int_82, hop_int_83, hop_int_84, hop_int_85, hop_int_86, hop_int_87, hop_int_88))))
