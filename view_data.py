from ast import Num
from glob import glob
from itertools import count
import os
import numpy as np
from pyparsing import nums
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

data_dir = {'train_data': './fewshotlogodetection_round1_train_202204/train/images/', 
            'train_json': './fewshotlogodetection_round1_train_202204/train/annotations/instances_train2017.json',
            'val_data': './fewshotlogodetection_round1_test_202204/val/images/',
            'val_json':'./fewshotlogodetection_round1_test_202204/val/annotations/instances_val2017.json'}

def data_summary():
    """
    training dataset and val dataset summary
    """
    train_list = glob(data_dir['train_data']+'*.jpg')
    val_list = glob(data_dir['val_data']+'*.jpg')
    print('train image counts:%d\nval image counts:%d\n'%(len(train_list),len(val_list)))

def view_train_shape():
    """
    view trainning dataset shape and visualization
    """
    sizes = []
    for img in glob(data_dir['val_data']+'*.jpg'):
        img = Image.open(img)
        sizes.append(img.size)
    sizes = np.array(sizes)
    # print(sizes)
    plt.figure(figsize=(20,16))
    plt.scatter(sizes[:,0],sizes[:,1])
    plt.xlabel('width')
    plt.ylabel('height')
    plt.title('image width-height summary')
    plt.show()


coco = COCO(annotation_file=data_dir['val_json'])

def view_dataset_categories():
    """
    view dataset categories
    """
    cats = coco.loadCats(coco.getCatIds())
    nms =[cat['name'] for cat in cats]
    print('coco categories sum:%d\n'%(len(nms)))
    print('coco categories:\n{}\n'.format('\t'.join(nms)))

def view_train_bbox():
    sizes =[]
    ids = list(sorted(coco.imgs.keys()))
    print('number of images :'.format(len(ids)))
    for img_id in ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        targets = coco.loadAnns(ann_ids)
        for target in targets:
            x,y,h,w = target["bbox"]
            sizes.append([h,w])
            
    sizes = np.array(sizes)    
    plt.figure(figsize=(20,16))
    plt.scatter(sizes[:,0],sizes[:,1])
    plt.xlabel('width')
    plt.ylabel('height')
    plt.title('bbox width-height summary')
    print(sizes.size)
    plt.show()

    

view_train_shape()
view_train_bbox()
# data_summary()
view_dataset_categories()
