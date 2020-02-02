import tensorflow as tf
import time
from . import help
from . import flow
from .ops import op_create, identity
from .ops import HEADER, LINE
from .framework import create_framework
from ..dark.darknet import Darknet
import json
import os

class TFNet(object):

	_TRAINER = dict({
		'rmsprop': tf.train.RMSPropOptimizer,
		'adadelta': tf.train.AdadeltaOptimizer,
		'adagrad': tf.train.AdagradOptimizer,
		'adagradDA': tf.train.AdagradDAOptimizer,
		'momentum': tf.train.MomentumOptimizer,
		'adam': tf.train.AdamOptimizer,
		'ftrl': tf.train.FtrlOptimizer,
		'sgd': tf.train.GradientDescentOptimizer
	})

	# imported methods
	_get_fps = help._get_fps
	say = help.say
	train = flow.train
	camera = help.camera
	predict = flow.predict
	return_predict = flow.return_predict
	to_darknet = help.to_darknet
	build_train_op = help.build_train_op
	load_from_ckpt = help.load_from_ckpt

	def __init__(self, FLAGS, darknet = None):
		self.ntrain = 0

		if isinstance(FLAGS, dict):
			from ..defaults import argHandler
			newFLAGS = argHandler()
			newFLAGS.setDefaults()
			newFLAGS.update(FLAGS)#将FLAGS里的键值对添加、覆盖到newFLAGS里
			FLAGS = newFLAGS#若从cli.py里调用，则是冗余操作；在其他地方调用TFNet()，则不是

		self.FLAGS = FLAGS
		if self.FLAGS.pbLoad and self.FLAGS.metaLoad:#当变量为''或None时，为Flase。如果从pb中加载
			self.say('\nLoading from .pb and .meta')
			self.graph = tf.Graph()#新建graph
			device_name = FLAGS.gpuName \
				if FLAGS.gpu > 0.0 else None
			with tf.device(device_name):#加载graph
				with self.graph.as_default() as g:
					self.build_from_pb()
			return

		if darknet is None:	
			darknet = Darknet(FLAGS)#layers保存层的对象list，但层的对象只是保存对应参数，不涉及tf网络
			self.ntrain = len(darknet.layers)

		self.darknet = darknet
		args = [darknet.meta, FLAGS]
		self.num_layer = len(darknet.layers)
		self.framework = create_framework(*args)#相比于darknet.meta，添加了labels和对应的color
		
		self.meta = darknet.meta

		self.say('\nBuilding net ...')
		start = time.time()
		self.graph = tf.Graph()#新建graph
		device_name = FLAGS.gpuName \
			if FLAGS.gpu > 0.0 else None
		with tf.device(device_name):
			with self.graph.as_default() as g:
				self.build_forward()#将权重转换为tf类型，建立tf网络
				self.setup_meta_ops()#判断采用的设备，是否保存，训练相关的op等
		self.say('Finished in {}s\n'.format(
			time.time() - start))
	
	def build_from_pb(self):
		with tf.gfile.FastGFile(self.FLAGS.pbLoad, "rb") as f:
			graph_def = tf.GraphDef()
			graph_def.ParseFromString(f.read())
		
		tf.import_graph_def(
			graph_def,
			name=""
		)#加载graph_def到默认graph
		with open(self.FLAGS.metaLoad, 'r') as fp:
			self.meta = json.load(fp)
		self.framework = create_framework(self.meta, self.FLAGS)

		# Placeholders
		self.inp = tf.get_default_graph().get_tensor_by_name('input:0')
		self.feed = dict() # other placeholders
		self.out = tf.get_default_graph().get_tensor_by_name('output:0')
		
		self.setup_meta_ops()
	
	def build_forward(self):
		verbalise = self.FLAGS.verbalise

		# Placeholders 类似于形參  #从此处开始，每一个BaseOp对象的inp，out都是tf型的
		inp_size = [None] + self.meta['inp_size']
		self.inp = tf.placeholder(tf.float32, inp_size, 'input')#tf形參
		self.feed = dict() # other placeholders

		# Build the forward pass
		state = identity(self.inp)#相当于多了一个input层，新建BaseOp对象的out等于此处的inp；
									#在第一个crop生成tf时，verbalise会额外输出input的信息
		roof = self.num_layer - self.ntrain
		self.say(HEADER, LINE)
		for i, layer in enumerate(self.darknet.layers):
			scope = '{}-{}'.format(str(i),layer.type)
			args = [layer, state, i, roof, self.feed]#传入到BaseOp类的初始化中
			state = op_create(*args)#将保存参数的对象lay里的权重转化成命名空间里的tf变量,并建立对应的网络层
			mess = state.verbalise()#获取当前转化的信息
			self.say(mess)
		self.say(LINE)

		self.top = state
		self.out = tf.identity(state.out, name='output')

	def setup_meta_ops(self):
		cfg = dict({
			'allow_soft_placement': False,
			'log_device_placement': True
		})
		#判断采用cpu还是gpu，以及使用gpu的比例
		utility = min(self.FLAGS.gpu, 1.)
		if utility > 0.0:
			self.say('GPU mode with {} usage'.format(utility))
			cfg['gpu_options'] = tf.GPUOptions(
				per_process_gpu_memory_fraction = utility)
			cfg['allow_soft_placement'] = True
		else: 
			self.say('Running entirely on CPU')
			cfg['device_count'] = {'GPU': 0}

		if self.FLAGS.train: self.build_train_op()#计算loss和梯度，得到train_op=apply梯度
		
		if self.FLAGS.summary:
			self.summary_op = tf.summary.merge_all()
			self.writer = tf.summary.FileWriter(self.FLAGS.summary + 'train')
		
		self.sess = tf.Session(config = tf.ConfigProto(**cfg))
		self.sess.run(tf.global_variables_initializer())

		if not self.ntrain: return
		self.saver = tf.train.Saver(tf.global_variables(), 
			max_to_keep = self.FLAGS.keep)#实例化save对象，setup_meta_ops、cli.py在train的时候会调用save，
		if self.FLAGS.load != 0: self.load_from_ckpt()
		
		if self.FLAGS.summary:
			self.writer.add_graph(self.sess.graph)#保存静态图

	def savepb(self):
		"""
		Create a standalone const graph def that 
		C++	can load and run.
		"""
		darknet_pb = self.to_darknet()
		flags_pb = self.FLAGS
		flags_pb.verbalise = False
		
		flags_pb.train = False
		# rebuild another tfnet. all const.
		tfnet_pb = TFNet(flags_pb, darknet_pb)		
		tfnet_pb.sess = tf.Session(graph = tfnet_pb.graph)
		# tfnet_pb.predict() # uncomment for unit testing
		name = 'built_graph/{}.pb'.format(self.meta['name'])
		os.makedirs(os.path.dirname(name), exist_ok=True)
		#Save dump of everything in meta
		with open('built_graph/{}.meta'.format(self.meta['name']), 'w') as fp:
			json.dump(self.meta, fp)
		self.say('Saving const graph def to {}'.format(name))
		graph_def = tfnet_pb.sess.graph_def
		tf.train.write_graph(graph_def,'./', name, False)