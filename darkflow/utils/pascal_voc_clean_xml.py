"""
parse PASCAL VOC xml annotations
"""

import os
import sys
import xml.etree.ElementTree as ET
import glob


def _pp(l): # pretty printing 
    for i in l: print('{}: {}'.format(i,l[i]))

def pascal_voc_clean_xml(ANN, pick, exclusive = False):# pick是标签
    print('Parsing for {} {}'.format(
            pick, 'exclusively' * int(exclusive)))

    dumps = list()
    cur_dir = os.getcwd()
    os.chdir(ANN)#改变当前目录到指定目录
    annotations = os.listdir('.')#返回当前目录(指定路径)下的文件和文件夹列表,无序随机
    annotations = glob.glob(str(annotations)+'*.xml')#返回所有匹配的文件路径列表,str(annotations)同时检测子文件夹
    size = len(annotations)

    for i, file in enumerate(annotations):#迭代每个xml文件
        # progress bar 进度条     
        sys.stdout.write('\r')
        percentage = 1. * (i+1) / size
        progress = int(percentage * 20)
        bar_arg = [progress*'=', ' '*(19-progress), percentage*100]
        bar_arg += [file]
        sys.stdout.write('[{}>{}]{:.0f}%  {}'.format(*bar_arg))
        sys.stdout.flush()#打印之前缓存的write信息
        
        # actual parsing 
        in_file = open(file)
        tree=ET.parse(in_file)#读取xml文件，转化为tree结构
        root = tree.getroot()
        jpg = str(root.find('filename').text)#root.find('filename')返回的是一个对象
        imsize = root.find('size')
        w = int(imsize.find('width').text)
        h = int(imsize.find('height').text)
        all = list()

        for obj in root.iter('object'):#迭代<annotation>下的每个<object>
                current = list()
                name = obj.find('name').text
                if name not in pick:
                        continue

                xmlbox = obj.find('bndbox')
                xn = int(float(xmlbox.find('xmin').text))
                xx = int(float(xmlbox.find('xmax').text))
                yn = int(float(xmlbox.find('ymin').text))
                yx = int(float(xmlbox.find('ymax').text))
                current = [name,xn,yn,xx,yx]
                all += [current]

        add = [[jpg, [w, h, all]]]#[图片名_1,[图片w,图片h,[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]]
        dumps += add
        in_file.close()

    # gather all stats
    stat = dict()
    for dump in dumps:
        all = dump[1][2]
        for current in all:
            if current[0] in pick:
                if current[0] in stat:
                    stat[current[0]]+=1
                else:
                    stat[current[0]] =1

    print('\nStatistics:')
    _pp(stat)#对所有框进行统计
    print('Dataset size: {}'.format(len(dumps)))

    os.chdir(cur_dir)
    return dumps#保存[图片名_1,[图片w,图片h,[目标名_1,目标框xmin,目标框ymin,目标框xmax,目标框ymax]]的列表