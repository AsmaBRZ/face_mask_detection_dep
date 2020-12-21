import torch
import h5py
import json
import numpy as np
from PIL import Image
from skimage.transform import resize
from flask import jsonify 
import os
import matplotlib
import base64

model_w = None
device = 'cpu'

def make_image(image, objects):
    fig,ax = plt.subplots(1)
    fig.patch.set_visible(False)
    
    plt.axis('off')
 
    ax.imshow(image)

    for annotation in objects:
      xmin, ymin, xmax, ymax = annotation['bbox']
      rect = patches.Rectangle((xmin,ymin), (xmax-xmin), (ymax-ymin), linewidth=1, edgecolor=classes_color[annotation['name']], facecolor='none')
      ax.add_patch(rect)
    #extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #plt.savefig(root_folder+'inference.png',bbox_inches=extent )
    #plt.show()
    encoded = fig_to_base64(fig)
    return '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

    

def predict(data):  
    global model_w
    device=torch.device('cpu')

    if model_w is None:
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/model/model.pth"
        model_w = torch.load(my_path)
        model.eval()
        
    normalize = T.Compose([T.ToTensor()])
    classes_color = {'with_mask':'g', 'without_mask':'r', 'mask_weared_incorrect':'tab:orange'}

    image = Image.open(data).convert('RGB')
    or_image=image
    image=F.resize(image, (64,64))
    image=normalize(image)
    preds = model([image])[0]
    keep = torchvision.ops.nms(preds['boxes'], preds['scores'], 0.00001)
    resized_obj = enlargeBB(preds,or_image.size[0],or_image.size[1])
    new_pred_boxes = new_objects_nms(resized_obj,keep)

    obj = { 'pred':make_image(or_image, new_pred_boxes)}
    
    return json.dumps(obj)
