from ..utils.process import cfg_yielder
from .darkop import create_darkop
from ..utils import loader
import warnings
import time
import os

class Darknet(object):

    _EXT = '.weights'

    def __init__(self, FLAGS):
        self.get_weight_src(FLAGS)#将模型权重的path赋值给self.src_bin;将模型配置的path赋值给self.src_cfg.
        self.modify = False

        print('Parsing {}'.format(self.src_cfg))
        #将.cfg里的 crop+网络主干 生成对应的对象，对象保存对应参数，对象list存入layers里;将模型的一些具体参数 存入字典meta里
        src_parsed = self.parse_cfg(self.src_cfg, FLAGS)
        self.src_meta, self.src_layers = src_parsed
        
        if self.src_cfg == FLAGS.model:
            self.meta, self.layers = src_parsed
        else: 
        	print('Parsing {}'.format(FLAGS.model))
        	des_parsed = self.parse_cfg(FLAGS.model, FLAGS)
        	self.meta, self.layers = des_parsed

        self.load_weights()#向layers里的加载权重，layers保存层的对象list，但层的对象只是保存对应参数，不涉及tf网络

    def get_weight_src(self, FLAGS):
        """
        将模型权重的path赋值给self.src_bin;
        将模型配置的path赋值给self.src_cfg.
        analyse FLAGS.load to know where is the 
        source binary and what is its config.
        can be: None, FLAGS.model, or some other
        """
        self.src_bin = FLAGS.model + self._EXT
        self.src_bin = FLAGS.binary + self.src_bin
        self.src_bin = os.path.abspath(self.src_bin)
        exist = os.path.isfile(self.src_bin)

        if FLAGS.load == str(): FLAGS.load = int()
        if type(FLAGS.load) is int:
            self.src_cfg = FLAGS.model
            if FLAGS.load: self.src_bin = None
            elif not exist: self.src_bin = None
        else:
            assert os.path.isfile(FLAGS.load), \
            '{} not found'.format(FLAGS.load)
            self.src_bin = FLAGS.load
            name = loader.model_name(FLAGS.load)#模型的名字
            cfg_path = os.path.join(FLAGS.config, name + '.cfg')
            if not os.path.isfile(cfg_path):
                warnings.warn(
                    '{} not found, use {} instead'.format(
                    cfg_path, FLAGS.model))
                cfg_path = FLAGS.model
            self.src_cfg = cfg_path
            FLAGS.load = int()


    def parse_cfg(self, model, FLAGS):
        """
        将.cfg里的 crop+网络主干 生成对应的对象，保存对应参数;将模型的一些具体参数 存入字典meta里
        return a list of `layers` objects (darkop.py)
        given path to binaries/ and configs/
        """
        args = [model, FLAGS.binary]
        cfg_layers = cfg_yielder(*args)#生成迭代器
        meta = dict(); layers = list()
        for i, info in enumerate(cfg_layers):#工厂模式 生成对应的对象，但此时的对象只是保存一些参数
            if i == 0: meta = info; continue
            else: new = create_darkop(*info)#info为Layer或其子类的初始化参数：type、num、传入到Layer.setup函数的参数
            layers.append(new)
        return meta, layers

    def load_weights(self):
        """
        Use `layers` and Loader to load .weights file
        """
        print('Loading {} ...'.format(self.src_bin))
        start = time.time()

        args = [self.src_bin, self.src_layers]
        #加载权重，重新生成需要参数的那些层对应的对象，保存到vals中，并将weight文件的数据加载到vals对应的层中
        wgts_loader = loader.create_loader(*args)
        for layer in self.layers: layer.load(wgts_loader)
        
        stop = time.time()
        print('Finished in {}s'.format(stop - start))