from cgi import print_arguments
from glob import glob
from random import random
from matplotlib import image
import mmcv
import matplotlib.pyplot as plt
from PIL import Image
import glob
import seaborn as sns
import numpy as np
from pycocotools.coco import COCO
import pandas as pd
from prettytable import PrettyTable

data_dir = {
    'train_data': './fewshotlogodetection_round1_train_202204/train/images',
    'val_data': '',
    'train_json': '',
    'val_json': './fewshotlogodetection_round1_train_202204/train/annotations/instances_train2017.json'
}

img_path = glob.glob(data_dir['train_data'] + '/*')
coco = COCO(annotation_file=data_dir['val_json'])


def get_img_size(img_path):
    """
    获取图片尺寸，并打印图片数量
    """
    image = Image.open(img_path)
    weight, height = image.size
    return (weight, height)

def view_tiny_bbox(coco):
    """
    查看小尺寸bbox在各类图片中的比例
    """
    class_names = []
    wh_ratios_cls = []
    for cat_id in coco.cats:
        wh_ratios = []
        for ann_id in coco.getAnnIds(catIds=[cat_id]):
            ann = coco.anns[ann_id]
            image_id = ann['image_id']
            w_ratio = ann['bbox'][2] / coco.imgs[image_id]['width']
            h_ratio = ann['bbox'][3] / coco.imgs[image_id]['height']
            wh_ratios.append([w_ratio, h_ratio])
        wh_ratios = np.array(wh_ratios)
        wh_ratios[:, -1] = wh_ratios[:, 0] * wh_ratios[:, 1]
        class_names.append(coco.cats[cat_id]['name'])
        wh_ratios_cls.append(
            (wh_ratios[:, -1] < 0.03).sum() / wh_ratios.shape[0])
    return [class_names,wh_ratios_cls]


def view_bbox_scale(coco):
    """
    获取bbox的长宽比
    """
    bbox_wh_ratios = []
    for cat_id in coco.cats:
        for ann_id in coco.getAnnIds(catIds=[cat_id]):
            ann = coco.anns[ann_id]
        wh_ratios = round(
            max(ann['bbox'][2], ann['bbox'][3]) /
            min(ann['bbox'][2], ann['bbox'][3]))
        bbox_wh_ratios.append(wh_ratios)    
    return bbox_wh_ratios


def draw_table(w_list, h_list, bbox_wh_ratios):
    """
    绘制数据图标，分别为：图片宽度统计、图片高度统计
                    图片长宽比统计、bbox长宽比统计
    """
    f,ax = plt.subplots(2, 2, figsize=(16, 8))
    f1 = sns.distplot(w_list, ax=ax[0, 0], kde=False, rug=False)
    f2 = sns.distplot(h_list, ax=ax[0, 1], kde=False, rug=False)
    f3 = sns.distplot(np.array(w_list) / np.array(h_list),
                    ax=ax[1, 0],
                    kde=False,
                    rug=False)
    f4 = sns.distplot(bbox_wh_ratios,ax=ax[1,1],kde=False,rug=False)
    f1.set_title('image width summary')
    f2.set_title('image height summary')
    f3.set_title('image width/height summary')
    f4.set_title('bbox scale ratio summary')
    plt.show()

def print_tiny_table(class_names,wh_ratios):
    """
    打印输出小bbox在各类图片中的比例
    """
    table_data = PrettyTable()
    table_data.add_column('class', class_names)
    table_data.add_column('small object ratio', wh_ratios)
    print(table_data.get_string())

def print_bbox_table(bbox_wh_ratios):
    """
    打印输出bbox长宽比的统计数量
    """
    table_bbox_hw_dic = {}
    for item in bbox_wh_ratios:
        if item in table_bbox_hw_dic: table_bbox_hw_dic[item] += 1
        else: table_bbox_hw_dic[item] = 1
    table_bbox_hw = PrettyTable()
    table_bbox_hw.add_column('bbox scale ratio',
                            list(table_bbox_hw_dic.keys()))
    table_bbox_hw.add_column('sum', list(table_bbox_hw_dic.values()))
    print(table_bbox_hw)


results = mmcv.track_parallel_progress(get_img_size, img_path, 16)
w_list = [x[0] for x in results]
h_list = [x[1] for x in results]
img_sum = len(img_path)
print('\n==============image sum:%d=============='%img_sum)
class_names = view_tiny_bbox(coco)[0]
bbox = view_tiny_bbox(coco)[1]
bbox_scale_ratio = view_bbox_scale(coco)
print_tiny_table(class_names,bbox)
print_bbox_table(bbox_scale_ratio)
draw_table(w_list,h_list,bbox_scale_ratio)
# print(class_names)




