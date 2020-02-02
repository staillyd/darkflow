import tensorflow.contrib.slim as slim
import pickle
import tensorflow as tf
from .misc import show
import numpy as np
import os

def loss(self, net_out):
    """
    Takes net.out and placeholders value
    returned in batch() func above,
    to build train_op and loss
    """
    # meta
    m = self.meta
    sprob = float(m['class_scale'])#yolo1:1
    sconf = float(m['object_scale'])#yolo1:1
    snoob = float(m['noobject_scale'])#yolo1:0.5
    scoor = float(m['coord_scale'])#yolo1:5
    S, B, C = m['side'], m['num'], m['classes']
    SS = S * S # number of grid cells

    print('{} loss hyper-parameters:'.format(m['model']))
    print('\tside    = {}'.format(m['side']))
    print('\tbox     = {}'.format(m['num']))
    print('\tclasses = {}'.format(m['classes']))
    print('\tscales  = {}'.format([sprob, sconf, snoob, scoor]))

    size1 = [None, SS, C]
    size2 = [None, SS, B]

    # return the below placeholders
    _probs = tf.placeholder(tf.float32, size1)#(?,SS,C) 目标中心点所在网格 对应的 目标类别概率为1。其他都为0
    _confs = tf.placeholder(tf.float32, size2)#(?,SS,B) 网格是否有目标(目标中心点所在网格所有box 对应的置信度为1。其他为0)
    _coord = tf.placeholder(tf.float32, size2 + [4])#(?,SS,B,4) 目标中心点所在网格所有box对应的 目标框坐标信息
    #_coord:[在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]

    # weights term for L2 loss
    _proid = tf.placeholder(tf.float32, size1)#yolo2
    # material calculating IOU
    _areas = tf.placeholder(tf.float32, size2) #目标中心点所在网格 对应的 目标框面积/网格面积;其余为0
    _upleft = tf.placeholder(tf.float32, size2 + [2])#目标中心点所在网格 对应的 目标框左上/网格;其余为0
    _botright = tf.placeholder(tf.float32, size2 + [2])#目标中心点所在网格 对应的 目标框右下/网格;其余为0

    self.placeholders = {#在flow的train()里使用
        'probs':_probs, 'confs':_confs, 'coord':_coord, 'proid':_proid,
        'areas':_areas, 'upleft':_upleft, 'botright':_botright
    }

    # Extract the coordinate prediction from net.out
    coords = net_out[:, SS * (C + B):]
    coords = tf.reshape(coords, [-1, SS, B, 4])#预测[在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # unit: grid cell  #预测的 目标框w/网格w,目标框h/网格h
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # unit: grid cell^2  #预测的 目标框面积/网格面积
    centers = coords[:,:,:,0:2] # [batch, SS, B, 2] #预测的目标中心点 在网格里占网格w的比例,占网格h的比例
    floor = centers - (wh * .5) # [batch, SS, B, 2] #预测box的 左、上边/网格w_h
    ceil  = centers + (wh * .5) # [batch, SS, B, 2] #预测box的 右、下边/网格w_h

    # calculate the intersection areas  size为[batch, SS, B]
    intersect_upleft   = tf.maximum(floor, _upleft)
    intersect_botright = tf.minimum(ceil , _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1])#预测box与真实box的交集面积

    # calculate the best IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas + area_pred - intersect)#点除 [batch, SS, B]
    #tf.reduce_max(iou, [2], True):max(iou[:,:,i]) True:维数保持不变 [batch,SS,1]
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))#[batch, SS, B] 在每样本每网格最大iou的box为true否则false
    best_box = tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)#每个样本存在目标的网格iou最大的box为1其他为0 [每样本每网格最大iou的box为true否则false * 网格是否有目标(目标中心点所在网格所有box 对应的置信度为1。其他为0)]

    # take care of the weight terms
    conid = snoob * (1. - confs) + sconf * confs #函数中与C_i相关的系数
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3)#(batch,SS,B,4)
    cooid = scoor * weight_coo #函数中与X_i相关的系数
    proid = sprob * _proid#yolo2 

    # flatten 'em all  网络输出拟合的是下面的真值
    probs = slim.flatten(_probs)#真值 _(?,SS,C)目标中心点所在网格 对应的 目标类别概率为1。其他都为0
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)#_(?,SS,B)每个样本存在目标的网格iou最大的box为1其他为0
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)#真值 _(?,SS,B,4) 目标中心点所在网格所有box对应的 目标框坐标信息;其他网格为0[在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
    cooid = slim.flatten(cooid)

    self.fetch += [probs, confs, conid, cooid, proid]
    true = tf.concat([probs, confs, coord], 1)
    wght = tf.concat([proid, conid, cooid], 1)
    print('Building {} loss'.format(m['model']))
    loss = tf.pow(net_out - true, 2)
    loss = tf.multiply(loss, wght)
    loss = tf.reduce_sum(loss, 1)
    self.loss = .5 * tf.reduce_mean(loss)
    tf.summary.scalar('{} loss'.format(m['model']), self.loss)
