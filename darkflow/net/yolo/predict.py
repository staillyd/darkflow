from ...utils.im_transform import imcv2_recolor, imcv2_affine_trans
from ...utils.box import BoundBox, box_iou, prob_compare
import numpy as np
import cv2
import os
import json
from ...cython_utils.cy_yolo_findboxes import yolo_box_constructor

def _fix(obj, dims, scale, offs):#obj:[目标名,目标框xmin,目标框ymin,目标框xmax,目标框ymax]
	for i in range(1, 5):
		dim = dims[(i + 1) % 2]#图像原shape (w,h,c)
		off = offs[(i + 1) % 2]#offs:[对放大图形的初始裁减w坐标,h坐标]
		obj[i] = int(obj[i] * scale - off)#原图像对图像缩放，裁减到原大小 im=(im*scale)[off:off+dim]
		obj[i] = max(min(obj[i], dim), 0)#将原图像的目标框坐标转化到缩放后的图像坐标

def resize_input(self, im):#im:cv2读取的(h,w,c)
	h, w, c = self.meta['inp_size']
	imsz = cv2.resize(im, (w, h))#cv2里的resize和其他cv2函数不同，输入参数是(resize_w,resize_h)型  imsz(resize_h,resize_w,c)
	imsz = imsz / 255.#归一化
	imsz = imsz[:,:,::-1]#cv2读取的是gbr,转换成rgb形式
	return imsz

def process_box(self, b, h, w, threshold):
	max_indx = np.argmax(b.probs)
	max_prob = b.probs[max_indx]
	label = self.meta['labels'][max_indx]
	if max_prob > threshold:
		left  = int ((b.x - b.w/2.) * w)#box里是归一化后的(中心x,y，矩形w,h)
		right = int ((b.x + b.w/2.) * w)
		top   = int ((b.y - b.h/2.) * h)
		bot   = int ((b.y + b.h/2.) * h)
		if left  < 0    :  left = 0
		if right > w - 1: right = w - 1
		if top   < 0    :   top = 0
		if bot   > h - 1:   bot = h - 1
		mess = '{}'.format(label)
		return (left, right, top, bot, mess, max_indx, max_prob)
	return None

def findboxes(self, net_out):
	meta, FLAGS = self.meta, self.FLAGS
	threshold = FLAGS.threshold
	
	boxes = []
	boxes = yolo_box_constructor(meta, net_out, threshold)
	
	return boxes

def preprocess(self, im, allobj = None):#im是一幅图像的path
	"""
	Takes an image, return it as a numpy tensor that is readily
	to be fed into tfnet. If there is an accompanied annotation (allobj),
	meaning this preprocessing is serving the train process, then this
	image will be transformed with random noise to augment training data,
	using scale, translation, flipping and recolor. The accompanied
	parsed annotation (allobj) will also be modified accordingly.
	"""
	if type(im) is not np.ndarray:#将图像转化为numpy形式
		im = cv2.imread(im)#cv里的是(h,w,c)

	#增强训练数据
	if allobj is not None: # in training mode
		result = imcv2_affine_trans(im)#对图像缩放，裁减到原大小，抛硬币确定是否水平翻转
		im, dims, trans_param = result#im:缩放翻转后的图像、dims:原图像shape(w,h,c)
		scale, offs, flip = trans_param#scale:图像缩放比例、offs:[对放大图形的初始裁减w坐标,h坐标]、flip:是否翻转
		for obj in allobj:#对训练数据的每个目标框[目标名,目标框xmin,目标框ymin,目标框xmax,目标框ymax]
			_fix(obj, dims, scale, offs)#将原图像的目标框坐标转化到缩放后的图像坐标
			if not flip: continue#将原图像的目标框坐标转化到水平翻转后的图像坐标
			obj_1_ =  obj[1]
			obj[1] = dims[0] - obj[3]
			obj[3] = dims[0] - obj_1_
		im = imcv2_recolor(im)#重新上色,添加随机噪声

	im = self.resize_input(im)#resize为网络输入的shape->归一化->bgr转为rgb im:(resize_h,resize_h,c)
	if allobj is None: return im
	return im#, np.array(im) # for unit testing

def postprocess(self, net_out, im, save = True):
	"""
	Takes net output, draw predictions, save to disk
	"""
	meta, FLAGS = self.meta, self.FLAGS
	threshold = FLAGS.threshold
	colors, labels = meta['colors'], meta['labels']

	boxes = self.findboxes(net_out)#cython里的函数，返回多个box，每个包含(中心x,y比例，矩形w,h比例)、每一类的概率、c

	if type(im) is not np.ndarray:
		imgcv = cv2.imread(im)
	else: imgcv = im

	h, w, _ = imgcv.shape
	resultsForJSON = []
	for b in boxes:
		boxResults = self.process_box(b, h, w, threshold)
		if boxResults is None:
			continue
		left, right, top, bot, mess, max_indx, confidence = boxResults#confidence是当前box所有类 confidence 的最大值
		thick = int((h + w) // 300)
		if self.FLAGS.json:
			resultsForJSON.append({"label": mess, "confidence": float('%.2f' % confidence), "topleft": {"x": left, "y": top}, "bottomright": {"x": right, "y": bot}})
			continue

		cv2.rectangle(imgcv,#画框
			(left, top), (right, bot),
			self.meta['colors'][max_indx], thick)
		cv2.putText(#写文字
			imgcv, mess, (left, top - 12),
			0, 1e-3 * h, self.meta['colors'][max_indx],
			thick // 3)


	if not save: return imgcv

	outfolder = os.path.join(self.FLAGS.imgdir, 'out')
	img_name = os.path.join(outfolder, os.path.basename(im))
	if self.FLAGS.json:
		textJSON = json.dumps(resultsForJSON)
		textFile = os.path.splitext(img_name)[0] + ".json"
		with open(textFile, 'w') as f:
			f.write(textJSON)
		return	

	cv2.imwrite(img_name, imgcv)#保存画框后的图片
