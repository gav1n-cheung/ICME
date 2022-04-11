from cgi import print_arguments
import os
import random
import numpy as np
from torch import randint
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image,ImageDraw
import matplotlib.pyplot as plt

json_path = './fewshotlogodetection_round1_train_202204/train/annotations/instances_train2017.json'
img_path = './fewshotlogodetection_round1_train_202204/train/images'

coco = COCO(annotation_file=json_path)

ids = list(sorted(coco.imgs.keys()))
print("number of images :{}".format(len(ids)))

coco_classes = dict([(v["id"],v["name"]) for k,v in coco.cats.items()])
print(coco_classes)

for img_id in ids[:]:
    ann_ids = coco.getAnnIds(imgIds=img_id)
    targets = coco.loadAnns(ann_ids)
    path = coco.loadImgs(img_id)[0]['file_name']
    print(path)
    img = Image.open(os.path.join(img_path,path)).convert('RGB')
    draw = ImageDraw.Draw(img)
    for target in targets:
        x,y,w,h = target["bbox"]
        x1,y1,x2,y2 = x,y,int(x+w),int(y+h)
        draw.rectangle((x1,y1,x2,y2),outline=tuple(np.random.randint(0,255,size=[3])))
        draw.text((x1,y1),coco_classes[target['category_id']].encode("utf-8").decode("latin1"))
    plt.imshow(img)
    plt.show()
