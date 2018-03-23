from shared.utilities import *

import cv2
import numpy as np
import json

"""{'supercategory': 'person', 'id': 1, 'name': 'person'}
{'supercategory': 'vehicle', 'id': 2, 'name': 'bicycle'}
{'supercategory': 'vehicle', 'id': 3, 'name': 'car'}
{'supercategory': 'vehicle', 'id': 4, 'name': 'motorcycle'}
{'supercategory': 'vehicle', 'id': 5, 'name': 'airplane'}
{'supercategory': 'vehicle', 'id': 6, 'name': 'bus'}
{'supercategory': 'vehicle', 'id': 7, 'name': 'train'}
{'supercategory': 'vehicle', 'id': 8, 'name': 'truck'}
{'supercategory': 'vehicle', 'id': 9, 'name': 'boat'}
{'supercategory': 'outdoor', 'id': 10, 'name': 'traffic light'}
{'supercategory': 'outdoor', 'id': 11, 'name': 'fire hydrant'}
{'supercategory': 'outdoor', 'id': 13, 'name': 'stop sign'}
{'supercategory': 'outdoor', 'id': 14, 'name': 'parking meter'}
{'supercategory': 'outdoor', 'id': 15, 'name': 'bench'}
{'supercategory': 'animal', 'id': 16, 'name': 'bird'}
{'supercategory': 'animal', 'id': 17, 'name': 'cat'}
{'supercategory': 'animal', 'id': 18, 'name': 'dog'}
{'supercategory': 'animal', 'id': 19, 'name': 'horse'}
{'supercategory': 'animal', 'id': 20, 'name': 'sheep'}
{'supercategory': 'animal', 'id': 21, 'name': 'cow'}"""

def hnm(args):
    path_anno = '../data/coco/annotations_trainval2017/annotations/instances_val2017.json'
    path_source = '../data/coco/val2017/val2017'
    path_target = '../data/face/hnm/coco'

    anno = None
    with open(path_anno, 'r') as f:
        anno = json.load(f)

    def get_image_by_id(id):
        for ele in anno['images']:
            if ele['id']==id:
                return ele
        return None

    def get_anno_by_image_id(id):
        _annos = list()
        for ele in anno['annotations']:
            if ele['image_id']==id:
                _annos.append(ele)
        return _annos

    for ele in anno['images']:
        id = ele['id']
        file_name = ele['file_name']
        _annos = get_anno_by_image_id(id) 
        print(_annos)

        inpath = path_source + '/' + file_name
        img = cv2.imread(inpath, 1)

        for _object in _annos:
            bbox = _object['bbox']
            bbox = np.array(bbox, dtype=np.int).tolist()
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

        canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
        ViewportManager().put('preview', img, (0, 0))
        ViewportManager().update('preview')
        
        k = ViewportManager().wait_key()
        if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
            pass
        
    return
    
    categories = dict()
    for ele in anno['categories']:
        categories[ele['id']] = ele

    print()
    count = 2000
    for ele in anno['annotations']:
        category_id = ele['category_id']
        image_id = ele['image_id']
        cate = categories[category_id]
        anno_img = get_image_by_id(image_id)
        if anno_img is not None:
            bbox = ele['bbox']
            print(cate)
            print()

            inpath = path_source + '/' + anno_img['file_name']
            img = cv2.imread(inpath, 1)
            bbox = np.array(bbox, dtype=np.int).tolist()
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

            canvas = ViewportManager().open('preview', shape=img.shape, blocks=(1, 2))
            ViewportManager().put('preview', img, (0, 0))
            ViewportManager().update('preview')
            
            k = ViewportManager().wait_key()
            if k in (ViewportManager.KEY_ENTER, ViewportManager.KEY_SPACE):
                pass

        count -= 1
        if count<=0: break