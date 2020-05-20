import matplotlib.pyplot as plt
from collections import namedtuple
import json

from image_metrics.edgeMetrics import *
from image_metrics.blobMetrics import *
from image_metrics.standard_metrics import *
from image_metrics.parameters import *



def tuplify(x):
    if type(x) is list:
        x = tuple(x)
    return x 


class Measurement:
    def __init__(self, name, fn, params=None):
        self.name = name
        self.fn = fn
        self.params = params
        self.data = []

    def run(self, img1, img2):
        params = ()
        if self.params:
            params = tuple([getattr(self.params,x) for x in self.params._fields])
            ret = self.fn(img1, img2, params)
        else:
            ret = self.fn(img1, img2)

        self.data.append(ret)

class Metric:

    def __init__(self, file_path = './config.json'):
        
        self.metrics = {}
        
        json_file = open(file_path)
        js = json.load(json_file)

        for config in js['Config']:
            include = eval(config['Include'])
            param_name = config['Name']
            if config['Parameters'] != 'None' and include:
                param_tuple = namedtuple("Parameters", [p for p in config['Parameters'].keys()])
                
                vals = list(config['Parameters'].values()) 
                vals = list(map(tuplify, vals))

                params = param_tuple(*tuple(vals))
            else: 
                params = None
            if include: 
                self.metrics[param_name] = Measurement(param_name, eval(config['Function']), params)


    def run(self, origImgs, cmpImgs):
        #assert origImgs.shape is cmpImgs.shape

        for o,c in zip(origImgs, cmpImgs):
            for k  in self.metrics:
                self.metrics[k].run(o, c)
 

    def showValues(self):
        pass
