import numpy as np

kernels_all = [[] for i in range(5)]
num_cycle1 = [1, 2, 3, 4, 5]  

kernels_all2 = [[] for i in range(7)]
num_cycle2 = [1, 2, 3, 4, 5, 6, 7] 

kernels_all3 = [[] for i in range(1)]

kernels_all4 = [[] for i in range(1)]

def GenerateKernels():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle1: 
        kernels = []
        for j in range(i):  
            k_size = (2 * i) + 1  #卷积核尺寸
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32)  
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2 #红区左上角坐标
            red_size = (j + 1) * 2 - 1#中心红区尺寸
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size #红区赋正值，和为1
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val #红区赋值
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) 
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val 

            kernels.append(kernel)
        kernels_all[i - 1] = kernels
        pass
    return kernels_all#生成5个卷积核，每个卷积核有i个kernel

def GenerateKernels2():
    """
    生成固定权值卷积核
    :return: None
    """
    for i in num_cycle2:  
        kernels = []
        for j in range(1): 
            k_size = (2 * i) + 1 
            kernel = np.zeros(shape=(k_size, k_size)).astype(np.float32) 
            lt_y = lt_x = k_size // 2 - ((j + 1) * 2 - 1) // 2  
            red_size = (j + 1) * 2 - 1 
            red_val = 1 / kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size  
            kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size] = red_val  
            blue_val = -1 / (k_size ** 2 - kernel[lt_x:lt_x + red_size, lt_y:lt_y + red_size].size) #蓝区赋负值，和为-1
            kernel[0:lt_x, :] = kernel[lt_x + red_size:, :] = kernel[:, :lt_y] = kernel[:, lt_y + red_size:] = blue_val #蓝区赋值 
            kernels.append(kernel)
        kernels_all2[i - 1] = kernels
        pass
    return kernels_all2#生成5个卷积核，每个卷积核有1个kernel

def GenerateKernels3():#生成一个 3×3 的均值卷积核，并将其存入
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 9.0
    kernels_all3[0].append(kernel)
    return kernels_all3

def GenerateKernels4():
    kernel = np.ones(shape=(3, 3)).astype(np.float32)
    kernel = kernel / 8.0 * -1
    kernel[1, 1] = 0
    kernels_all4[0].append(kernel)
    return kernels_all4
