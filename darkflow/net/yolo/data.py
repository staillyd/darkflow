from ...utils.pascal_voc_clean_xml import pascal_voc_clean_xml
from numpy.random import permutation as perm
from .predict import preprocess
# from .misc import show
from copy import deepcopy
import pickle
import numpy as np
import os 

def parse(self, exclusive = False):
    meta = self.meta
    ext = '.parsed'
    ann = self.FLAGS.annotation
    if not os.path.isdir(ann):#如果没ann，则退出
        msg = 'Annotation directory not found {} .'
        exit('Error: {}'.format(msg.format(ann)))
    print('\n{} parsing {}'.format(meta['model'], ann))
    #解析voc数据集里ann的xml文件
    dumps = pascal_voc_clean_xml(ann, meta['labels'], exclusive)
    return dumps#保存[图片名_1,[图片w,图片h,[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]]的列表


def _batch(self, chunk):#传入1个[图片名,[图片w,图片h,[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]]
    """
    Takes a chunk of parsed annotations
    returns value for placeholders of net's 
    input & loss layer correspond to this chunk
    """
    meta = self.meta
    S, B = meta['side'], meta['num']
    C, labels = meta['classes'], meta['labels']

    # preprocess
    jpg = chunk[0]; w, h, allobj_ = chunk[1]
    allobj = deepcopy(allobj_)#保存[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]的列表
    path = os.path.join(self.FLAGS.dataset, jpg)#传入图片的路径
    img = self.preprocess(path, allobj)#增强训练数据，同时将原数据对应的ann转化到增强后的坐标;resize到网络输入shape;归一化;gbr->rgb.
    #img:(resize_h,resize_w,c)

    # Calculate regression target
    cellx = 1. * w / S
    celly = 1. * h / S#原图像分割为S*S个网格，每个网格的宽高
    for obj in allobj:#对训练数据的每个目标框[目标名,目标框xmin,目标框ymin,目标框xmax,目标框ymax]
        centerx = .5*(obj[1]+obj[3]) #xmin, xmax #目标的中心点_w
        centery = .5*(obj[2]+obj[4]) #ymin, ymax #目标的中心点_h
        cx = centerx / cellx#(目标中心点在w轴上属于第cx个网格).(在网格里占网格w的比例)
        cy = centery / celly#(目标中心点在h轴上属于第cy个网格).(在网格里占网格y的比例)
        if cx >= S or cy >= S: return None, None
        obj[3] = float(obj[3]-obj[1]) / w#[目标名,目标框xmin,目标框ymin,目标框w/图像w,目标框ymax]
        obj[4] = float(obj[4]-obj[2]) / h#[目标名,目标框xmin,目标框ymin,目标框w/图像w,目标框h/图像h]
        obj[3] = np.sqrt(obj[3])#[目标名,目标框xmin,目标框ymin,sqrt(目标框w/图像w),目标框h/图像h]
        obj[4] = np.sqrt(obj[4])#[目标名,目标框xmin,目标框ymin,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
        obj[1] = cx - np.floor(cx) # centerx #[目标名,在网格里占网格w的比例,目标框ymin,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
        obj[2] = cy - np.floor(cy) # centery #[目标名,在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h)]
        obj += [int(np.floor(cy) * S + np.floor(cx))]#加上 目标中心点在第几个网格的编码 
        #!!!np.floor(cy)相当于网格像素h坐标。np.floor(cx)相当于网格像素w坐标。
        #在Test输出时(cy_yolo_findboxes.pyx)，w横着一排一排扫。所以这里是int(np.floor(cy) * S + np.floor(cx))!!!
    # allobj为保存[目标名,在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h),目标中心点在第几个网格的编码]的列表
    # show(im, allobj, S, w, h, cellx, celly) # unit test

    # Calculate placeholders' values 真值设定  !!存在问题:如果两个目标的中心点 都在一个网格里只会算一个目标的真值
    probs = np.zeros([S*S,C])#真实概率  每个网格预测每类的概率
    confs = np.zeros([S*S,B])#网格是否有目标(目标中心点所在网格所有box 对应的置信度为1;其余为0)
    coord = np.zeros([S*S,B,4])#每个网格的每个box的坐标信息
    proid = np.zeros([S*S,C])#yolo2
    prear = np.zeros([S*S,4])#目标中心点所在网格 对应的 目标框边界线坐标信息
    for obj in allobj:#[目标名,在网格里占网格w的比例,占网格h的比例,sqrt(目标框w/图像w),sqrt(目标框h/图像h),目标中心点在第几个网格的编码]
        probs[obj[5], :] = [0.] * C
        probs[obj[5], labels.index(obj[0])] = 1.#目标中心点所在网格 对应的 目标类别概率为1。其他都为0
        proid[obj[5], :] = [1] * C#yolo2
        coord[obj[5], :, :] = [obj[1:5]] * B#目标中心点所在网格所有box对应的 目标框坐标信息
        prear[obj[5],0] = obj[1] - obj[3]**2 * .5 * S # xleft  目标中心点所在网格 对应的 目标框w_min/网格w
        prear[obj[5],1] = obj[2] - obj[4]**2 * .5 * S # yup    目标中心点所在网格 对应的 目标框h_min/网格h
        prear[obj[5],2] = obj[1] + obj[3]**2 * .5 * S # xright 目标中心点所在网格 对应的 目标框w_max/网格w
        prear[obj[5],3] = obj[2] + obj[4]**2 * .5 * S # ybot   目标中心点所在网格 对应的 目标框h_max/网格h;其他网格都为0
        confs[obj[5], :] = [1.] * B#目标中心点所在网格所有box 对应的置信度为1;其余为0

    # Finalise the placeholders' values
    upleft   = np.expand_dims(prear[:,0:2], 1)#添加一个第二维(SS,1,2) 目标中心点所在网格 对应的 目标框左上/网格 信息;其余为0
    botright = np.expand_dims(prear[:,2:4], 1)#添加一个第二维(SS,1,2) 目标中心点所在网格 对应的 目标框右下/网格 信息;其余为0
    wh = botright - upleft; 
    area = wh[:,:,0] * wh[:,:,1]
    upleft   = np.concatenate([upleft] * B, 1)#(SS,B,2) 目标中心点所在网格 对应的 目标框左上/网格 信息;其余为0
    botright = np.concatenate([botright] * B, 1)#(SS,B,2) 目标中心点所在网格 对应的 目标框右下/网格 信息;其余为0
    areas = np.concatenate([area] * B, 1)#(SS,B,2) 目标中心点所在网格 对应的 目标框面积/网格面积 信息;其余为0

    # value for placeholder at input layer
    inp_feed_val = img#(h,w,c)
    # value for placeholder at loss layer 
    loss_feed_val = {
        'probs': probs, 'confs': confs, 
        'coord': coord, 'proid': proid,
        'areas': areas, 'upleft': upleft, 
        'botright': botright
    }

    return inp_feed_val, loss_feed_val

def shuffle(self):
    batch = self.FLAGS.batch#每批次的数据量
    data = self.parse()#从指定ann文件夹中解析所有xml文件
    #data保存[图片名_1,[图片w,图片h,[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]]的列表
    size = len(data)#所有训练数据的长度

    print('Dataset of {} instance(s)'.format(size))
    if batch > size: self.FLAGS.batch = batch = size
    batch_per_epoch = int(size / batch)#整体训练集分成batch_per_epoch批次

    for i in range(self.FLAGS.epoch):#整体训练次数
        shuffle_idx = perm(np.arange(size))#打乱数据顺序
        for b in range(batch_per_epoch):#整体训练集分成batch_per_epoch批次
            # yield these
            x_batch = list()
            feed_batch = dict()

            for j in range(b*batch, b*batch+batch):#每批次的数据
                train_instance = data[shuffle_idx[j]]
                try:
                    inp, new_feed = self._batch(train_instance)#inp:当前这张图片的数据(h,w,c) new_feed:从ann得到的tf输入
                except ZeroDivisionError:
                    print("This image's width or height are zeros: ", train_instance[0])
                    print('train_instance:', train_instance)
                    print('Please remove or fix it then try again.')
                    raise

                if inp is None: continue
                x_batch += [np.expand_dims(inp, 0)]#x_batch:[?,h,w,c]

                for key in new_feed:
                    new = new_feed[key]
                    old_feed = feed_batch.get(key, 
                        np.zeros((0,) + new.shape))
                    feed_batch[key] = np.concatenate([ 
                        old_feed, [new] 
                    ])#将这批次的feed数据拼接    
            
            x_batch = np.concatenate(x_batch, 0)#(?,h,w,c)
            yield x_batch, feed_batch
        
        print('Finish {} epoch(es)'.format(i + 1))

