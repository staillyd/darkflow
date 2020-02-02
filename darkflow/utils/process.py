"""
WARNING: spaghetti code.
"""

import numpy as np
import pickle
import os

def parser(model):
	"""
	将.cfg里的 crop+网络主干 的参数 存入字典layers里;将模型的一些具体参数 存入字典meta里
	Read the .cfg file to extract layers into `layers`
	as well as model-specific parameters into `meta`
	"""
	def _parse(l, i = 1):
		return l.split('=')[i].strip()

	with open(model, 'rb') as f:
		lines = f.readlines()

	lines = [line.decode() for line in lines]	
	
	meta = dict(); layers = list() # will contains layers' info
	h, w, c = [int()] * 3; layer = dict()
	for line in lines:
		line = line.strip()
		line = line.split('#')[0]
		if '[' in line:
			if layer != dict(): #在cfg第二个以及之后的[]时
				if layer['type'] == '[net]': #如果上一个结构是[net]
					h = layer['height']
					w = layer['width']
					c = layer['channels']
					meta['net'] = layer #将[net]结构保存到meta['net']中
				else:#上一个结构不是[net]时
					if layer['type'] == '[crop]': #如果上一个结构是[crop]
						h = layer['crop_height']
						w = layer['crop_width']
					layers += [layer]				
			layer = {'type': line}
		else:
			try:#如果当前行右侧是数字，转化成数字，得到(key,val)
				i = float(_parse(line))
				if i == int(i): i = int(i)
				layer[line.split('=')[0].strip()] = i
			except:#否则直接转化为(key,val)
				try:
					key = _parse(line, 0)
					val = _parse(line, 1)
					layer[key] = val
				except:#空白行
					'banana ninja yadayada'

	meta.update(layer) # last layer contains meta info
	if 'anchors' in meta:
		splits = meta['anchors'].split(',')
		anchors = [float(x.strip()) for x in splits]
		meta['anchors'] = anchors
	meta['model'] = model # path to cfg, not model name
	meta['inp_size'] = [h, w, c] #保存crop之后的[h,w,c]
	return layers, meta

def cfg_yielder(model, binary):
	"""
	yielding each layer information to initialize `layer`
	"""
	layers, meta = parser(model) #将.cfg里的 crop+网络主干 的参数 存入字典layers里;将模型的一些具体参数 存入字典meta里
	yield meta
	h, w, c = meta['inp_size']; l = w * h * c#裁减后的h,w,c  即输入到主干网络的h,w,c

	# Start yielding
	flat = False # flag for 1st dense layer
	conv = '.conv.' in model
	for i, d in enumerate(layers):
		#-----------------------------------------------------
		if d['type'] == '[crop]':
			yield ['crop', i]#裁减，新建一个Layer对象
		#-----------------------------------------------------
		elif d['type'] == '[local]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			activation = d.get('activation', 'logistic')
			w_ = (w - 1 - (1 - pad) * (size - 1)) // stride + 1
			h_ = (h - 1 - (1 - pad) * (size - 1)) // stride + 1
			yield ['local', i, size, c, n, stride, 
					pad, w_, h_, activation]
			if activation != 'linear': yield [activation, i]
			w, h, c = w_, h_, n
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[convolutional]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			yield ['convolutional', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation]#卷积，新建一个convolutional_layer对象
			if activation != 'linear': yield [activation, i]#激活层，新建一个Layer对象
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			w, h, c = w_, h_, n#当前结构的输出size，也是下一个结构的输入size
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[maxpool]':
			stride = d.get('stride', 1)
			size = d.get('size', stride)
			padding = d.get('padding', (size-1) // 2)
			yield ['maxpool', i, size, stride, padding]#池化层，新建一个maxpool对象
			w_ = (w + 2*padding) // d['stride'] 
			h_ = (h + 2*padding) // d['stride']
			w, h = w_, h_#当前结构的输出size，也是下一个结构的输入size
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[avgpool]':
			flat = True; l = c
			yield ['avgpool', i]
		#-----------------------------------------------------
		elif d['type'] == '[softmax]':
			yield ['softmax', i, d['groups']]
		#-----------------------------------------------------
		elif d['type'] == '[connected]':
			if not flat:#迭代器里记录变量，相当于只拉直输入到第一个全连接层的神经元
				yield ['flatten', i]#拉直，新建一个Layer对象
				flat = True
			activation = d.get('activation', 'logistic')
			#此时的l是当前层的输入，d['output']是cfg文件的参数，迭代器会记录变量
			yield ['connected', i, l, d['output'], activation]
			if activation != 'linear': yield [activation, i]#激活层，新建一个Layer对象
			l = d['output']#当前结构输出的size
		#-----------------------------------------------------
		elif d['type'] == '[dropout]': 
			yield ['dropout', i, d['probability']]#dropout层，新建一个dropout_layer对象
		#-----------------------------------------------------
		elif d['type'] == '[select]':
			if not flat:
				yield ['flatten', i]
				flat = True
			inp = d.get('input', None)
			if type(inp) is str:
				file = inp.split(',')[0]
				layer_num = int(inp.split(',')[1])
				with open(file, 'rb') as f:
					profiles = pickle.load(f, encoding = 'latin1')[0]
				layer = profiles[layer_num]
			else: layer = inp
			activation = d.get('activation', 'logistic')
			d['keep'] = d['keep'].split('/')
			classes = int(d['keep'][-1])
			keep = [int(c) for c in d['keep'][0].split(',')]
			keep_n = len(keep)
			train_from = classes * d['bins']
			for count in range(d['bins']-1):
				for num in keep[-keep_n:]:
					keep += [num + classes]
			k = 1
			while layers[i-k]['type'] not in ['[connected]', '[extract]']:
				k += 1
				if i-k < 0:
					break
			if i-k < 0: l_ = l
			elif layers[i-k]['type'] == 'connected':
				l_ = layers[i-k]['output']
			else:
				l_ = layers[i-k].get('old',[l])[-1]
			yield ['select', i, l_, d['old_output'],
				   activation, layer, d['output'], 
				   keep, train_from]
			if activation != 'linear': yield [activation, i]
			l = d['output']
		#-----------------------------------------------------
		elif d['type'] == '[conv-select]':
			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			d['keep'] = d['keep'].split('/')
			classes = int(d['keep'][-1])
			keep = [int(x) for x in d['keep'][0].split(',')]

			segment = classes + 5
			assert n % segment == 0, \
			'conv-select: segment failed'
			bins = n // segment
			keep_idx = list()
			for j in range(bins):
				offset = j * segment
				for k in range(5):
					keep_idx += [offset + k]
				for k in keep:
					keep_idx += [offset + 5 + k]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			c_ = len(keep_idx)
			yield ['conv-select', i, size, c, n, 
				   stride, padding, batch_norm,
				   activation, keep_idx, c_]
			w, h, c = w_, h_, c_
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[conv-extract]':
			file = d['profile']
			with open(file, 'rb') as f:
				profiles = pickle.load(f, encoding = 'latin1')[0]
			inp_layer = None
			inp = d['input']
			out = d['output']
			inp_layer = None
			if inp >= 0:
				inp_layer = profiles[inp]
			if inp_layer is not None:
				assert len(inp_layer) == c, \
				'Conv-extract does not match input dimension'
			out_layer = profiles[out]

			n = d.get('filters', 1)
			size = d.get('size', 1)
			stride = d.get('stride', 1)
			pad = d.get('pad', 0)
			padding = d.get('padding', 0)
			if pad: padding = size // 2
			activation = d.get('activation', 'logistic')
			batch_norm = d.get('batch_normalize', 0) or conv
			
			k = 1
			find = ['[convolutional]','[conv-extract]']
			while layers[i-k]['type'] not in find:
				k += 1
				if i-k < 0: break
			if i-k >= 0:
				previous_layer = layers[i-k]
				c_ = previous_layer['filters']
			else:
				c_ = c
			
			yield ['conv-extract', i, size, c_, n, 
				   stride, padding, batch_norm,
				   activation, inp_layer, out_layer]
			if activation != 'linear': yield [activation, i]
			w_ = (w + 2 * padding - size) // stride + 1
			h_ = (h + 2 * padding - size) // stride + 1
			w, h, c = w_, h_, len(out_layer)
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[extract]':
			if not flat:
				yield['flatten', i]
				flat = True
			activation = d.get('activation', 'logistic')
			file = d['profile']
			with open(file, 'rb') as f:
				profiles = pickle.load(f, encoding = 'latin1')[0]
			inp_layer = None
			inp = d['input']
			out = d['output']
			if inp >= 0:
				inp_layer = profiles[inp]
			out_layer = profiles[out]
			old = d['old']
			old = [int(x) for x in old.split(',')]
			if inp_layer is not None:
				if len(old) > 2: 
					h_, w_, c_, n_ = old
					new_inp = list()
					for p in range(c_):
						for q in range(h_):
							for r in range(w_):
								if p not in inp_layer:
									continue
								new_inp += [r + w*(q + h*p)]
					inp_layer = new_inp
					old = [h_ * w_ * c_, n_]
				assert len(inp_layer) == l, \
				'Extract does not match input dimension'
			d['old'] = old
			yield ['extract', i] + old + [activation] + [inp_layer, out_layer]
			if activation != 'linear': yield [activation, i]
			l = len(out_layer)
		#-----------------------------------------------------
		elif d['type'] == '[route]': # add new layer here
			routes = d['layers']
			if type(routes) is int:
				routes = [routes]
			else:
				routes = [int(x.strip()) for x in routes.split(',')]
			routes = [i + x if x < 0 else x for x in routes]
			for j, x in enumerate(routes):
				lx = layers[x]; 
				xtype = lx['type']
				_size = lx['_size'][:3]
				if j == 0:
					h, w, c = _size
				else: 
					h_, w_, c_ = _size
					assert w_ == w and h_ == h, \
					'Routing incompatible conv sizes'
					c += c_
			yield ['route', i, routes]
			l = w * h * c
		#-----------------------------------------------------
		elif d['type'] == '[reorg]':
			stride = d.get('stride', 1)
			yield ['reorg', i, stride]
			w = w // stride; h = h // stride; 
			c = c * (stride ** 2)
			l = w * h * c
		#-----------------------------------------------------
		else:
			exit('Layer {} not implemented'.format(d['type']))

		#保存到layers里了，d为layers的元素，按地址操作
		d['_size'] = list([h, w, c, l, flat])#当前结构的输出size，也是下一个结构的输入size

	if not flat: meta['out_size'] = [h, w, c]
	else: meta['out_size'] = l