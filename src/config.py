'''Set configurations for all the raw and processed datasets
Takes in path to raw data, tracks, processed df, and then later interim data
'''
import os
from os.path import join as oj
import numpy as np


# data paths
DIR_PROCESSED = '/scratch/users/vision/data/abc_data/data_processed/processed'
DIR_TRACKS = '/scratch/users/vision/data/abc_data/data_processed/tracks'

# raw data ################################################################################
data_dir_orig = '/scratch/users/vision/data/abc_data/auxilin_data_tracked'
d_new = '/scratch/users/vision/data/abc_data/auxilin_new_data/AI_Clathrin_molecularPrediction'
clath_aux_folder = 'CLTA-TagRFP+-+ EGFP-Aux1-A7D2+-+ EGFP-GAK-F6+-+  TIRF data'
dynamin_folder = 'CLTA-TagRFP EGFP-Aux1-GAK-F6 Dyn2-Halo-E1-JF646'


# data splitting
s_orig = 'A7D2'
s_orig_gak = 'EGFP-GAK F6'
s_clath_aux_no_a7d2 = 'CLTA-TagRFP EGFP-Aux1 EGFP-GAK F6' # goes [1-11]
s_clath_aux = 'CLTA-TagRFP EGFP-Aux1-A7D2 EGFP-GAK-F6' # goes [1, 2] + [5-12] - 13 is there but missing some tifs
s_a8 = 'CLTA-TagRFP EGFP-GAK A8'
s_dynamin = '488-1.5mW 561-1.5mW 647-1.5mW Exp100ms Int1.5s'
s_clath_pi4p = 'cme'
s_ap2_pi4p = 'cme'

DSETS = {
    'clath_aux+gak_a7d2': {  # this was the original data
        'data_dir': data_dir_orig,
        'feature_selection': np.array([f'{s_orig}/1']),
        'train': np.array([f'{s_orig}/{i}' for i in [1, 2, 3, 4, 5, 6]]),
        'test': np.array([f'{s_orig}/{i}' for i in [7, 8]])
    },
    'clath_aux+gak': {
        'data_dir': data_dir_orig,
        'feature_selection': np.array([f'{s_orig_gak}/1']),
        'train': np.array([f'{s_orig_gak}/{i}' for i in [1, 2, 3, 4, 5, 6]]),
        'test': np.array([f'{s_orig_gak}/{i}' for i in [7, 8]])
    },
    'clath_aux+gak_a7d2_new': {
        'data_dir': oj(d_new, clath_aux_folder),        
        'feature_selection': np.array([f'{s_clath_aux}/1']),
        'train': np.array([f'{s_clath_aux}/{i}' for i in [1, 2, 5, 6, 7, 8, 9, 10]]),
        'test': np.array([f'{s_clath_aux}/{i}' for i in [11, 12]])
    },
    'clath_aux_dynamin': {  # this is 3-channel data
        'data_dir': oj(d_new, dynamin_folder),        
        'feature_selection': None,
        'train': np.array([f'{s_dynamin}_{i}/1_1.5s' for i in [4, 5, 6]] + 
                          [f'{s_dynamin}_4_Pos{i}/1_1.5s' for i in np.arange(13) if i != 9]),
        'test': np.array([f'{s_dynamin}_7_Pos{i}/1_1.5s' for i in np.arange(6)]),
    },
    'clath_aux+gak_new': {
        'data_dir': oj(d_new, clath_aux_folder),
        'feature_selection': np.array([f'{s_clath_aux_no_a7d2}/1']),
        'train': np.array([f'{s_clath_aux_no_a7d2}/{i}' for i in [1, 2, 3, 4, 5, 6, 7, 9]]),
        'test': np.array([f'{s_clath_aux_no_a7d2}/{i}' for i in [10, 11]])
    }, 
    'clath_gak': {
        'data_dir': oj(d_new, 'CLTA-TagRFP+-+ EGFP-GAK-A8+-+  TIRF data'),
        'feature_selection': np.array([f'{s_a8}/1']),
        'train': np.array([f'{s_a8}/{i}' for i in [1, 2, 3, 4, 5, 6, 7]]),
        'test': np.array([f'{s_a8}/{i}' for i in [8, 9]])    
    },
    'clath_pi4p_notreatment': {
        'data_dir': oj(d_new, 'CLTA-TagRFP+-+ EGFP-DrrA-Aux1 PI4P probe TIRF data', 'notreatment'), 
        'feature_selection': np.array([f'{s_clath_pi4p}/2']),
        'train': np.array([f'{s_clath_pi4p}/{i}' for i in [2, 5]]),
        'test': np.array([f'{s_clath_pi4p}/{i}' for i in [9, 10]])  
    },
    'ap2_pi4p': {
        'data_dir': oj(d_new, 'AP2-TagRFP+-+ EGFP-DrrA-Aux1 PI4P probe TIRF data'),
        'feature_selection': np.array([f'{s_ap2_pi4p}/1']),
        'train': np.array([f'{s_ap2_pi4p}/{i}' for i in [1, 2, 4, 5, 6, 7, 8]]),
        'test': np.array([f'{s_ap2_pi4p}/{i}' for i in [9, 11]])  
    },
}


# gt labels for original data (by pid) ################################################################################

'''Note these labels constitute corrections to the y_consec_thresh defn (as of 02/09/2020)
'''
LABELS_ORIG = {
    'neg': [6725, 909, 5926, 983, 8224, 3363] +
           [221, 248, 255, 274, 291, 294, 350, 358, 364, 384, 434, 481, 518,
            535, 588, 590, 602, 604, 638, 668, 675, 708, 752, 778, 808, 827,
            852, 867, 884, 888, 897, 905, 952, 953, 967, 984, 987, 997, 1002,
            1003, 1015, 1030, 1045, 1051, 1055, 1070, 1091, 1100, 1144, 1178,
            1226, 1246, 1258, 1260, 1276, 1307, 1317, 1370, 1394, 1399, 1400,
            1423, 1432, 1451, 1613, 2115, 2143, 2286, 2330, 2369, 2381, 2399,
            2416, 2435, 2612, 2613, 2631, 2667, 2679, 2705, 2714, 2750, 2756,
            2857, 2962, 3002, 3017, 3032, 3061, 3128, 3181, 3216, 3801, 3819,
            3916, 3945, 3954, 3967, 3970, 4042, 4108, 4120, 4141, 4142, 4155,
            4162, 4165, 4172, 4221, 4279, 4290, 4297, 4328, 4333, 4341, 4384,
            4422, 4428, 4468, 4547, 4558, 4573, 4586, 4616, 4621, 4674, 4875,
            4936, 4941, 4947, 5151, 5391, 5602, 5693, 5706, 5707, 5946, 5973,
            6007, 6072, 6147, 6157, 6253, 6489, 6618, 7340, 7363, 7413, 7423,
            7424, 7455, 7483, 7485, 7512, 7565, 7568, 7641, 7647, 7690, 7728,
            7872, 7877, 7883, 7892, 7942, 7969, 7985, 8005, 8018, 8020, 8024,
            8111, 8120, 8166, 8255, 8264, 8375, 8462, 8484, 8512, 8532, 8829,
            8833, 8918, 8944] +
           [3964, 10054, 735, 846, 1362, 3823, 5389, 8834, 9565, 10882, 11166, 718] +
           [3964, 718, 735, 846, 1362, 3823, 5389, 8834, 9565, 10882, 11166, 10054],
    # note: first list were flipped, second list were already correct
    'pos': [3982, 8243, 777, 3940, 7559, 2455, 4748, 633, 2177, 1205, 603, 7972, 8458, 3041, 924, 8786, 4116, 885,
            6298, 4658, 7889, 982, 829, 1210, 3054, 504, 1164, 347, 627, 1470, 2662, 2813, 422, 8400, 7474, 1273,
            6365, 1559, 4348, 1156, 6250, 4864, 639, 930, 5424, 7818, 8463, 4358, 7656, 843, 890, 4373, 2737, 7524,
            2590, 3804, 7667, 2148, 8585, 2919, 5712, 4440, 1440, 4699, 1089, 3004, 3126, 2869, 4183, 7335, 3166,
            8461, 2180, 849, 6458, 4575, 4091, 3966, 4725, 2514, 7626, 3055, 4200, 6429, 1220, 4472, 8559, 412, 903,
            5440, 1084, 2136, 6833, 1189, 7521, 8141, 7939, 8421, 944, 1264, 298, 6600, 1309, 3043, 243, 4161, 6813,
            5464] +
           [238, 251, 389, 524, 556, 758, 759, 830, 1213, 1255, 1290, 1422, 1463, 1484, 1504, 2016, 2046,
            2061, 2077, 2083, 2106, 2112, 2116, 2124, 2129, 2135, 2195, 2209, 2217, 2234, 2273, 2291, 2313,
            2338, 2353, 2402, 2460, 2651, 2658, 2703, 2791, 2805, 2848, 3101, 3138, 3142, 3150, 3170, 3343,
            3367, 3918, 3946, 4223, 4386, 4430, 4583, 4608, 4620, 4724, 5366, 5383, 5384, 5407, 5412, 5415,
            5422, 5442, 5447, 5449, 5478, 5492, 5503, 5506, 5516, 5548, 5550, 5558, 5589, 5634, 5694, 5708,
            5728, 5760, 5780, 5787, 5788, 5800, 5811, 5813, 5814, 5879, 5882, 5885, 5888, 5891, 5899, 5911,
            5912, 5950, 5951, 5953, 5957, 5960, 5986, 5988, 6000, 6011, 6012, 6020, 6021, 6032, 6049, 6053,
            6065, 6096, 6106, 6113, 6118, 6123, 6152, 6155, 6202, 6237, 6246, 6248, 6263, 6266, 6272, 6273,
            6302, 6305, 6321, 6325, 6327, 6363, 6368, 6398, 6407, 6410, 6423, 6424, 6431, 6444, 6449, 6461,
            6462, 6482, 6490, 6517, 6518, 6526, 6568, 6586, 6594, 6601, 6608, 6640, 6656, 6662, 6683, 6684,
            6693, 6703, 6771, 6774, 6801, 6802, 6823, 6851, 7348, 7352, 7448, 7470, 7496, 7511, 7596, 7720,
            7787, 7805, 7819, 7826, 7885, 7900, 7908, 7926, 7930, 7951, 7965, 8000, 8072, 8109, 8122, 8123,
            8143, 8159, 8211, 8242, 8248, 8257, 8259, 8265, 8286, 8321, 8330, 8357, 8368, 8372, 8385, 8407,
            8430, 8436, 8444, 8448, 8454, 8490, 8507, 8513, 8556, 8604, 8639, 8750, 8751, 8755, 8764, 8777,
            8822, 8852, 8863, 8911, 8981, 2668, 889, 6066, 9529, 9676, 9990, 10157, 10183, 10243,
            10434] +
           [2321, 11032, 4484, 4750, 8084, 6770, 6624, 2749, 6378, 7833, 4399, 9547, 2253] +
           [2321, 10996, 4454, 10431, 11032, 10057, 4484, 8084, 10754, 2382, 938, 2228, 10887, 6770, 10895,
            10863, 6624, 10333, 4069, 10113, 4849, 9719, 10116, 245, 10077, 9547, 9557, 10457, 10037, 9900,
            10146, 5507, 10517, 2749, 9563, 6378, 2014, 9714, 1353, 10117, 7504, 9724, 3141, 5797, 10508, 10374,
            5593, 9932, 4399, 10632, 1039, 9904, 9930, 8505, 429, 10331, 5470, 8557, 7773, 10830, 10749, 2031,
            3822, 7833, 5791, 10602, 2203, 542, 10843, 7759, 10483, 4827, 225, 7679, 9617, 2378, 5409, 10142,
            9975, 10264, 918, 10148, 10066, 9917, 9485, 6400, 5961, 10023, 10418, 231, 10695, 3065, 6420, 7865,
            9813, 10765, 6290, 2270, 729, 2626, 8424, 10199, 2200, 2854, 2253],
    # note: first list were flipped, second list were already correct
    'hotspots': [6510, 6606, 2373, 6135, 6023, 7730, 2193, 8307, 5626, 4109, 2921, 4614, 2573, 7490, 6097, 7836,
                 1011, 6493, 5779, 8660, 6232, 6009, 2579, 929, 3824, 357, 6162, 477, 5640, 6467, 244, 2922, 4288,
                 2926, 1480, 4441, 4683, 8239, 9749, 9826, 9844, 10945, 11037] +
                [10069, 5485, 3146, 5560, 5600, 5937, 7688, 6055, 5670, 10235,
                 5583, 6151, 5720, 2553, 6040, 292, 5456, 2437, 5966, 5499, 10043,
                 10232, 5434, 6224, 5785, 6210, 2761, 6359, 6438, 5423, 5774,
                 7556, 5766, 7882, 7732, 5798, 2711, 2562, 5939, 2214, 2881,
                 2588, 10123, 6527, 10309, 2038, 2683, 5617, 2146, 4117, 10821,
                 2538, 5408, 5527, 6079, 7499, 6641, 2930, 5683, 6353, 5958, 2154, 5835] +
                [10015, 6339, 3168, 7481, 7779, 646, 4117, 7891, 4324, 2146, 10007, 1162,
                 10330, 285, 5527, 7616, 5617, 4196, 7771, 2085, 1104, 5512, 8303, 4409,
                 7343, 2538, 7570, 3977, 2683, 2038, 10250, 2494, 10309, 8423, 2417, 6353,
                 2564, 685, 6471, 6527, 10123, 2173, 2588, 205, 2881, 5863, 5958, 5408,
                 2607, 2214, 5939, 2562, 6079, 8208, 2154, 2799, 3909, 5798, 7663, 2574,
                 7732, 5835, 7882, 5766, 7556, 6438, 6641, 6359, 2761, 6210, 5434, 10232,
                 10043, 6109, 2323, 9550, 2437, 5456, 2930, 5683, 413, 2553, 7499, 6151,
                 5583, 10235, 6055, 7688, 5600, 5560, 3146, 5485, 10069, 5423, 5499, 6224, 5937, 292]
}

# neg should actually be neg, pos should actually be pos
LABELS_DYNAMIN = {
    'neg': [204777, 255597, 257856, 254414, 210090, 228965, 223120, 244787,
       210219, 205670, 207817, 208985, 243014, 244540, 243101, 205523,
       246994, 243813, 247387, 223499, 226986, 207715, 207461, 228022,
       246466, 246049, 249456, 249260, 243906, 248469, 248677, 246461,
       216470, 246446, 224584, 248959, 218106, 212688, 216575, 221288,
       246639, 213189, 250127, 216917, 239495, 218845, 249898, 248365,
       213853, 238928, 234087, 245726, 246329, 242735, 247350, 233576,
       243179, 242756, 247149, 249987, 248686, 245505, 245172, 237980,
       235687, 235962, 257613, 218102, 246146, 245821, 238930, 248793,
       243616, 243001, 242160, 229700, 222705, 224733, 214902, 205387,
       232491, 236402, 243552, 215134, 246401, 250057, 234118, 217468,
       243037, 239471, 217535, 249669, 235951, 238382, 237019, 208402,
       249106, 239288, 245737, 232361, 219713, 248476, 223001, 232697,
       231401, 215084, 252052, 214974, 205220, 257383, 219267, 212179,
       233113, 205274, 239585, 252200, 235908, 252770, 228385, 226748,
       218512, 242516, 256045, 252589, 238125, 222314] + \
       [258864, 228390, 207373, 205484, 207847, 258914], # from 40 worst errs
    'pos': [212256, 214854, 235817, 252489, 229125, 208935, 224734, 225681,
       257459, 220275, 237926, 207089, 237666, 215606, 252622, 227713,
       244095, 249564, 218127, 207014, 228064, 218584, 234681, 244427,
       205577, 221330, 222762, 259677, 217438, 227043, 239332, 225555,
       218725, 229340, 232956, 249714, 253432, 216647, 229976, 218328,
       243715, 205099, 252075, 216391, 218423, 215674, 223778, 238083,
       217819, 256932, 224605, 224576, 219263, 209148, 213457, 244520,
       219232, 208333, 228467, 228204, 255908, 249674, 213107, 221335,
       227820, 219138, 213488, 226788, 217890, 221283, 253437, 223868,
       246066, 238332, 256658, 210055, 230096, 204197, 212192, 205678,
       226214, 206266, 246347, 228949, 244897, 231391, 224139, 252028,
       219621, 253661, 245841, 214386, 222897, 225659, 247270, 204155,
       205157, 246161, 252953, 215964, 239835, 226266, 257059, 225921,
       211909, 228291, 211968, 224783, 246821, 249302, 224306, 247821,
       249017, 206983, 226609, 227327, 227044, 216729, 207840, 212872,
       225000, 228672, 234528, 225612, 234261, 214450, 212401, 236587,
       205938, 223768, 242149, 216730, 249207, 216074, 224790, 210343,
       212138, 253250, 213593, 235697, 235482, 237720, 217202, 233687,
       214668, 213157, 213751, 252837, 212942, 214825, 253023, 246366,
       234697] + \
       [238377, 216355, 212351, 244573, ], # from 40 worst errs
    'hotspots': [205537, 215233, 216903, 216241, 236608, 216941, 208591, 223088, # from tp
       217357, 214308, 237087, 247979, 207586, 216892, 217105, 213597,
       225263, 214879, 222906, 245501, 247460, 217602, 207269, 215196,
       216337, 245702, 224723, 245320, 213849, 209150, 226611, 248664,
       255361, 249998, 215481, 204719, 214674, 214432, 205126, 205146,
       218024, 206783, 225708, 247813, 255434, 208411, 212022, 212829,
       243765, 224915, 214699, 243846, 219117, 214503, 214622, 212061,
       208072, 243264, 213813, 228133, 224472, 247999, 213403, 208150,
       242500, 222445, 217523, 207271, 206599, 204323, 245793, 213013,
       218184, 216085, 236403, 208438, 246518, 225456, 216652, 245564,
       234101, 222997, 245241, 215456, 234109, 207612, 246110, 236069,
       204970, 225392, 235906, 212937, 234456, 246400, 225547, 213433,
       241137, 224283, 215557, 218920, 245648, 215118, 231402, 227704,
       235268, 242653, 254434, 244044, 229557, 246020, 204639, 223538,
       255904, 235393, 208086, 207861, 214203, 213260, 218675, 216595,
       237485, 204597, 239637, 212874, 246936, 243518, 235349, 254114,
       235877, 227335, 214540, 232583, 216441, 215495, 244976, 259687,
       204475, 224961, 233204, 218681, 242455, 239226, 223669, 214875,
       236654, 216784, 206824, 242479, 206267, 217824, 251958, 226253,
       214332] + \
       [242828, 254428, 257377, 207581, 235660, 205824, 225114, 213307, # from fn
       209487, 206981, 239467, 252051, 205115, 206939, 226274, 242758,
       228777, 256040, 255719, 235465, 209012, 222720, 256464, 223698,
       254836, 223231, 254747, 255906, 253835, 254133, 204951, 254969] + \
       [215974, 224389, 207496, 212337, 215555, 246347, 244847, 233344,
        244324, 242252], # from 40 worst errs
}


LABELS = {data_dir: None for data_dir in DSETS.keys()}
LABELS['clath_aux+gak_a7d2'] = LABELS_ORIG
LABELS['clath_aux_dynamin'] = LABELS_DYNAMIN



