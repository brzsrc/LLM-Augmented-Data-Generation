"""
cluster_config.py — 自动生成，勿手动修改
由 cluster_analysis.py 的 build_cluster_config() 生成
"""

CLUSTER_UIDS = {
    0: [1, 4, 7, 8, 9, 10, 11, 13, 16, 17, 18, 20, 21, 22, 25, 29, 30, 31, 34, 35, 36],
    1: [2, 12, 15, 23],
    2: [14, 26, 32, 33],
    3: [3],
}

CLUSTER_WEIGHTS = {0: 21, 1: 4, 2: 4, 3: 1}

CLUSTER_NAMES = {0: 'cluster_0_work', 1: 'cluster_1_university', 2: 'cluster_2_healthcare', 3: 'cluster_3_work'}

# key = (is_weekday: bool, slot: int)
CLUSTER_LOCATION_DIST = {
    0: {
        (False, 1): {'Home': 0.502347, 'Restaurant': 0.14554, 'Store': 0.098592, 'GroceryStore': 0.051643, 'Transit': 0.042254, 'Park': 0.037559, 'University': 0.028169, 'FurnitureStore': 0.023474, 'Unknown': 0.023474, 'Healthcare': 0.014085, 'Work': 0.014085, 'ClothingStore': 0.00939, 'Gym': 0.004695, 'ElectronicsStore': 0.004695},
        (False, 2): {'Home': 0.362745, 'Restaurant': 0.181373, 'Store': 0.166667, 'FurnitureStore': 0.053922, 'Park': 0.053922, 'Work': 0.044118, 'GroceryStore': 0.039216, 'Transit': 0.02451, 'University': 0.019608, 'Healthcare': 0.014706, 'Gym': 0.014706, 'ClothingStore': 0.014706, 'Unknown': 0.004902, 'ElectronicsStore': 0.004902},
        (False, 3): {'Home': 0.331818, 'Restaurant': 0.209091, 'Store': 0.131818, 'Work': 0.063636, 'Park': 0.063636, 'FurnitureStore': 0.045455, 'Transit': 0.040909, 'GroceryStore': 0.031818, 'ClothingStore': 0.027273, 'University': 0.022727, 'Gym': 0.018182, 'Healthcare': 0.009091, 'ElectronicsStore': 0.004545},
        (False, 4): {'Home': 0.351598, 'Restaurant': 0.191781, 'Store': 0.114155, 'Work': 0.086758, 'Park': 0.059361, 'Transit': 0.041096, 'ClothingStore': 0.041096, 'FurnitureStore': 0.03653, 'University': 0.022831, 'Healthcare': 0.018265, 'GroceryStore': 0.013699, 'Gym': 0.009132, 'ElectronicsStore': 0.009132, 'Unknown': 0.004566},
        (False, 5): {'Home': 0.319635, 'Restaurant': 0.237443, 'Store': 0.100457, 'Park': 0.082192, 'University': 0.050228, 'FurnitureStore': 0.045662, 'GroceryStore': 0.045662, 'Transit': 0.03653, 'ClothingStore': 0.027397, 'Work': 0.022831, 'ElectronicsStore': 0.009132, 'Healthcare': 0.009132, 'Gym': 0.009132, 'Unknown': 0.004566},
        (True, 1): {'Home': 0.460145, 'Restaurant': 0.101449, 'Work': 0.097826, 'University': 0.077899, 'Store': 0.077899, 'Park': 0.059783, 'Unknown': 0.030797, 'GroceryStore': 0.028986, 'Transit': 0.025362, 'FurnitureStore': 0.012681, 'Healthcare': 0.01087, 'ClothingStore': 0.009058, 'ElectronicsStore': 0.005435, 'Gym': 0.001812},
        (True, 2): {'Work': 0.412613, 'University': 0.181982, 'Home': 0.136937, 'Restaurant': 0.068468, 'Store': 0.041441, 'Healthcare': 0.032432, 'Park': 0.025225, 'GroceryStore': 0.021622, 'Transit': 0.01982, 'FurnitureStore': 0.018018, 'ClothingStore': 0.016216, 'Unknown': 0.016216, 'ElectronicsStore': 0.005405, 'Gym': 0.003604},
        (True, 3): {'Work': 0.460317, 'Home': 0.142857, 'University': 0.126984, 'Restaurant': 0.056437, 'Store': 0.045855, 'Park': 0.044092, 'Healthcare': 0.022928, 'Unknown': 0.022928, 'GroceryStore': 0.022928, 'FurnitureStore': 0.015873, 'ClothingStore': 0.014109, 'Transit': 0.014109, 'ElectronicsStore': 0.005291, 'Gym': 0.005291},
        (True, 4): {'Home': 0.315978, 'Work': 0.210054, 'Store': 0.10772, 'Restaurant': 0.096948, 'University': 0.055655, 'Park': 0.043088, 'FurnitureStore': 0.039497, 'GroceryStore': 0.035907, 'ClothingStore': 0.028725, 'Transit': 0.021544, 'Healthcare': 0.019749, 'Unknown': 0.014363, 'ElectronicsStore': 0.007181, 'Gym': 0.003591},
        (True, 5): {'Home': 0.376991, 'Restaurant': 0.153982, 'Store': 0.120354, 'University': 0.065487, 'Work': 0.051327, 'Park': 0.044248, 'ClothingStore': 0.035398, 'Transit': 0.031858, 'Healthcare': 0.031858, 'GroceryStore': 0.024779, 'Unknown': 0.023009, 'FurnitureStore': 0.021239, 'Gym': 0.010619, 'ElectronicsStore': 0.00885},
    },
    1: {
        (False, 1): {'ElectronicsStore': 0.659091, 'University': 0.204545, 'Restaurant': 0.068182, 'Store': 0.022727, 'Home': 0.022727, 'FurnitureStore': 0.022727},
        (False, 2): {'ElectronicsStore': 0.542857, 'University': 0.2, 'Store': 0.114286, 'FurnitureStore': 0.057143, 'Park': 0.057143, 'Home': 0.028571},
        (False, 3): {'ElectronicsStore': 0.428571, 'University': 0.214286, 'Healthcare': 0.047619, 'Park': 0.047619, 'Store': 0.047619, 'Restaurant': 0.047619, 'Home': 0.047619, 'ClothingStore': 0.02381, 'Transit': 0.02381, 'Work': 0.02381, 'GroceryStore': 0.02381, 'FurnitureStore': 0.02381},
        (False, 4): {'ElectronicsStore': 0.386364, 'University': 0.25, 'Restaurant': 0.159091, 'Store': 0.090909, 'FurnitureStore': 0.045455, 'Unknown': 0.022727, 'Work': 0.022727, 'Home': 0.022727},
        (False, 5): {'ElectronicsStore': 0.410256, 'University': 0.230769, 'Restaurant': 0.153846, 'GroceryStore': 0.051282, 'Park': 0.025641, 'Unknown': 0.025641, 'Store': 0.025641, 'FurnitureStore': 0.025641, 'Home': 0.025641, 'Healthcare': 0.025641},
        (True, 1): {'ElectronicsStore': 0.514019, 'University': 0.308411, 'Home': 0.121495, 'FurnitureStore': 0.028037, 'Transit': 0.018692, 'Unknown': 0.009346},
        (True, 2): {'University': 0.414894, 'Work': 0.212766, 'ElectronicsStore': 0.12766, 'Restaurant': 0.085106, 'Store': 0.074468, 'GroceryStore': 0.021277, 'Park': 0.021277, 'FurnitureStore': 0.021277, 'Home': 0.010638, 'ClothingStore': 0.010638},
        (True, 3): {'University': 0.357143, 'Work': 0.204082, 'ElectronicsStore': 0.163265, 'Store': 0.081633, 'FurnitureStore': 0.071429, 'Restaurant': 0.05102, 'Home': 0.020408, 'ClothingStore': 0.020408, 'Healthcare': 0.010204, 'Park': 0.010204, 'Gym': 0.010204},
        (True, 4): {'University': 0.3, 'ElectronicsStore': 0.288889, 'Restaurant': 0.077778, 'Work': 0.066667, 'Store': 0.066667, 'Home': 0.066667, 'FurnitureStore': 0.055556, 'Park': 0.033333, 'Healthcare': 0.022222, 'ClothingStore': 0.022222},
        (True, 5): {'ElectronicsStore': 0.384615, 'University': 0.252747, 'Restaurant': 0.098901, 'Store': 0.054945, 'FurnitureStore': 0.054945, 'Park': 0.032967, 'Home': 0.032967, 'Healthcare': 0.021978, 'Unknown': 0.021978, 'Gym': 0.010989, 'ClothingStore': 0.010989, 'Transit': 0.010989, 'GroceryStore': 0.010989},
    },
    2: {
        (False, 1): {'Home': 0.275, 'FurnitureStore': 0.25, 'Store': 0.175, 'ClothingStore': 0.15, 'GroceryStore': 0.1, 'Transit': 0.025, 'Park': 0.025},
        (False, 2): {'Store': 0.25, 'Restaurant': 0.15, 'FurnitureStore': 0.15, 'Park': 0.1, 'GroceryStore': 0.1, 'Home': 0.075, 'ClothingStore': 0.075, 'Healthcare': 0.05, 'Gym': 0.025, 'Transit': 0.025},
        (False, 3): {'Store': 0.318182, 'GroceryStore': 0.136364, 'FurnitureStore': 0.113636, 'ClothingStore': 0.113636, 'Home': 0.090909, 'Park': 0.068182, 'Transit': 0.068182, 'Restaurant': 0.068182, 'ElectronicsStore': 0.022727},
        (False, 4): {'Store': 0.306122, 'FurnitureStore': 0.163265, 'Restaurant': 0.122449, 'Home': 0.102041, 'ClothingStore': 0.102041, 'Transit': 0.061224, 'Park': 0.061224, 'GroceryStore': 0.040816, 'Healthcare': 0.020408, 'Unknown': 0.020408},
        (False, 5): {'Store': 0.363636, 'FurnitureStore': 0.181818, 'Restaurant': 0.159091, 'Home': 0.113636, 'ClothingStore': 0.068182, 'Transit': 0.045455, 'ElectronicsStore': 0.022727, 'Park': 0.022727, 'University': 0.022727},
        (True, 1): {'Healthcare': 0.373626, 'Restaurant': 0.142857, 'FurnitureStore': 0.10989, 'Home': 0.10989, 'Store': 0.076923, 'Transit': 0.043956, 'University': 0.043956, 'ClothingStore': 0.043956, 'Park': 0.032967, 'Gym': 0.010989, 'GroceryStore': 0.010989},
        (True, 2): {'Healthcare': 0.326531, 'Restaurant': 0.193878, 'Home': 0.142857, 'FurnitureStore': 0.102041, 'ClothingStore': 0.071429, 'Store': 0.05102, 'Gym': 0.040816, 'Park': 0.030612, 'University': 0.020408, 'GroceryStore': 0.010204, 'ElectronicsStore': 0.010204},
        (True, 3): {'Healthcare': 0.368932, 'Restaurant': 0.15534, 'Store': 0.126214, 'Home': 0.097087, 'FurnitureStore': 0.07767, 'ClothingStore': 0.058252, 'University': 0.038835, 'Gym': 0.029126, 'Transit': 0.019417, 'ElectronicsStore': 0.019417, 'Park': 0.009709},
        (True, 4): {'FurnitureStore': 0.2, 'Healthcare': 0.189474, 'Store': 0.178947, 'ClothingStore': 0.094737, 'Park': 0.084211, 'Restaurant': 0.073684, 'Home': 0.052632, 'Transit': 0.042105, 'University': 0.031579, 'GroceryStore': 0.021053, 'Gym': 0.021053, 'ElectronicsStore': 0.010526},
        (True, 5): {'Store': 0.376238, 'FurnitureStore': 0.178218, 'Home': 0.138614, 'Restaurant': 0.079208, 'ClothingStore': 0.079208, 'Transit': 0.059406, 'Healthcare': 0.029703, 'Park': 0.029703, 'GroceryStore': 0.009901, 'ElectronicsStore': 0.009901, 'Unknown': 0.009901},
    },
    3: {
        (False, 1): {'Gym': 0.7, 'Healthcare': 0.2, 'GroceryStore': 0.1},
        (False, 2): {'Gym': 0.727273, 'Park': 0.090909, 'Healthcare': 0.090909, 'GroceryStore': 0.090909},
        (False, 3): {'Gym': 0.416667, 'Park': 0.25, 'ClothingStore': 0.083333, 'Healthcare': 0.083333, 'University': 0.083333, 'Restaurant': 0.083333},
        (False, 4): {'Park': 0.307692, 'Gym': 0.153846, 'GroceryStore': 0.153846, 'University': 0.153846, 'ElectronicsStore': 0.076923, 'Store': 0.076923, 'Transit': 0.076923},
        (False, 5): {'Gym': 0.384615, 'Healthcare': 0.153846, 'Restaurant': 0.153846, 'Park': 0.076923, 'FurnitureStore': 0.076923, 'Store': 0.076923, 'GroceryStore': 0.076923},
        (True, 1): {'Work': 0.607143, 'University': 0.178571, 'Gym': 0.107143, 'Store': 0.035714, 'Park': 0.035714, 'Restaurant': 0.035714},
        (True, 2): {'Work': 0.538462, 'Restaurant': 0.192308, 'Gym': 0.076923, 'University': 0.076923, 'Park': 0.038462, 'Transit': 0.038462, 'GroceryStore': 0.038462},
        (True, 3): {'Work': 0.521739, 'Gym': 0.130435, 'Park': 0.086957, 'Store': 0.086957, 'University': 0.086957, 'Healthcare': 0.043478, 'Restaurant': 0.043478},
        (True, 4): {'Store': 0.16, 'Work': 0.16, 'Gym': 0.16, 'Transit': 0.12, 'FurnitureStore': 0.08, 'Park': 0.08, 'Healthcare': 0.08, 'Restaurant': 0.08, 'ClothingStore': 0.04, 'University': 0.04},
        (True, 5): {'Gym': 0.346154, 'Restaurant': 0.269231, 'Store': 0.076923, 'ClothingStore': 0.076923, 'Transit': 0.076923, 'GroceryStore': 0.038462, 'Park': 0.038462, 'Healthcare': 0.038462, 'University': 0.038462},
    },
}

CLUSTER_AVAIL_BY_SLOT = {
    0: {1: 0.9438, 2: 0.8748, 3: 0.8818, 4: 0.8157, 5: 0.8992},
    1: {1: 0.9801, 2: 0.8527, 3: 0.8714, 4: 0.8358, 5: 0.8538},
    2: {1: 0.9389, 2: 0.8116, 3: 0.8776, 4: 0.7639, 5: 0.8966},
    3: {1: 0.9737, 2: 0.9459, 3: 0.9714, 4: 0.5789, 5: 0.6667},
}
