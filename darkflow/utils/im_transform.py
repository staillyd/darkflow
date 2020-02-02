import numpy as np
import cv2

def imcv2_recolor(im, a = .1):
	t = [np.random.uniform()]
	t += [np.random.uniform()]
	t += [np.random.uniform()]
	t = np.array(t) * 2. - 1.#-1~1的三个随机数

	# random amplify each channel
	im = im * (1 + t * a)
	mx = 255. * (1 + a)
	up = np.random.uniform() * 2 - 1
# 	im = np.power(im/mx, 1. + up * .5)
	im = cv2.pow(im/mx, 1. + up * .5)
	return np.array(im * 255., np.uint8)

def imcv2_affine_trans(im):
	# Scale and translate
	h, w, c = im.shape#注意cv2的图像读取后是[h,w,c]
	scale = np.random.uniform() / 10. + 1.#1~1.1的一个随机数
	max_offx = (scale-1.) * w
	max_offy = (scale-1.) * h
	offx = int(np.random.uniform() * max_offx)#对放大图形的初始裁减w坐标
	offy = int(np.random.uniform() * max_offy)
	
	im = cv2.resize(im, (0,0), fx = scale, fy = scale)#图像缩放，scale=1~1.1。实际上是放大
	im = im[offy : (offy + h), offx : (offx + w)]#截取和原图像size一致的图形
	flip = np.random.binomial(1, .5)#抛硬币   #二项分布(总共实验次数,实验成功概率,size采样次数) 返回成功次数
	if flip: im = cv2.flip(im, 1)#水平翻转
	return im, [w, h, c], [scale, [offx, offy], flip]
