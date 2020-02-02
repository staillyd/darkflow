import numpy as np
cimport numpy as np
cimport cython
ctypedef np.float_t DTYPE_t
from libc.math cimport exp
from ..utils.box import BoundBox
from nms cimport NMS



@cython.cdivision(True)
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def yolo_box_constructor(meta,np.ndarray[float] net_out, float threshold):

    cdef:
        float sqrt
        int C,B,S
        int SS,prob_size,conf_size
        int grid, b
        int class_loop

    
    sqrt =  meta['sqrt'] + 1
    C, B, S = meta['classes'], meta['num'], meta['side']
    boxes = []
    SS        =  S * S # number of grid cells  7*7
    prob_size = SS * C # class probabilities   7*7*20
    conf_size = SS * B # confidences for each grid cell   7*7*2

    cdef:
        #网络输出的前7*7 *20个神经元用于概率预测  (SS,C) 真:目标中心点所在网格 对应的 目标类别概率为1。其他都为0
        #此时为每个网格预测包含物体的前提下每个类别的条件概率P(Class|Object)，每个网格只预测一组网格，与B无关
        float [:,::1] probs =  np.ascontiguousarray(net_out[0 : prob_size]).reshape([SS,C])
        #网络输出之后7*7 *2个神经元用于置信度预测  (SS,B) 真:每个样本存在目标的网格iou最大的box为1其他为0
        float [:,::1] confs =  np.ascontiguousarray(net_out[prob_size : (prob_size + conf_size)]).reshape([SS,B])
        #网络输出最后7*7 *2 *4个神经元用于坐标预测(总共7*7*30=1470个神经元)
        #真:(SS,B,4) 目标中心点所在网格所有box对应的 目标框坐标信息;其他网格为0
        #[在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
        float [: , : ,::1] coords =  np.ascontiguousarray(net_out[(prob_size + conf_size) : ]).reshape([SS, B, 4])
        float [:,:,::1] final_probs = np.zeros([SS,B,C],dtype=np.float32)
        
    # Test输出阶段
    for grid in range(SS):#将图像分成S*S个网格，对每个网格
        for b in range(B):#每个网格预测两个boxes，对每个box
            coords[grid, b, 0] = (coords[grid, b, 0] + grid %  S) / S#输出 框w像素坐标/w  
            #网络拟合的是obj[1]=框w像素坐标/网格w-网格像素的w坐标。grid%S:网格像素的w坐标。与data.py里obj[5]的编码对应。
            coords[grid, b, 1] = (coords[grid, b, 1] + grid // S) / S#输出b_h/h  b_w:框在图像上的像素坐标  h:图片高
            coords[grid, b, 2] =  coords[grid, b, 2] ** sqrt#目标框w/图像w
            coords[grid, b, 3] =  coords[grid, b, 3] ** sqrt#目标框h/图像h
            for class_loop in range(C):
                #输出概率为 (网格的条件概率)*每个box的置信度IoU
                probs[grid, class_loop] = probs[grid, class_loop] * confs[grid, b]
                #print("PROBS",probs[grid,class_loop])
                if(probs[grid,class_loop] > threshold ):#超过阈值时，输出概率
                    final_probs[grid, b, class_loop] = probs[grid, class_loop]
    
    
    return NMS(np.ascontiguousarray(final_probs).reshape(SS*B, C) , np.ascontiguousarray(coords).reshape(SS*B, 4))
