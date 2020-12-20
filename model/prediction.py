import torch
import h5py
import json
import numpy as np
from PIL import Image
from skimage.transform import resize
from flask import jsonify 
import os

import base64

model_w = None
device = 'cpu'

def plot_image(img_tensor, annotation):
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    for box in annotation["boxes"]:
        xmin, ymin, xmax, ymax = box

        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=1,edgecolor='r',facecolor='none')

        # Add the patch to the Axes
        ax.add_patch(rect)

    encoded = fig_to_base64(fig)
    return '<img src="data:image/png;base64, {}">'.format(encoded.decode('utf-8'))

def predict(data):  
    global model_w

    if model_w is None:
        txt=os.path.abspath(__file__)
        x = txt.split("/", 3)
        my_path="/"+x[1]+"/"+x[2]+"/model/model.pth"
        model_w = torch.load(my_path)
        

    CLASSES = ['background','with_mask','without_mask','mask_weared_incorrect']
    IMG_SHAPE = (64,64)
    image = Image.open(data)
    image=np.array(image)
    print(image.shape)
    
    image = image.astype('float32')
    image = resize(image, (32, 32), anti_aliasing=True)
    image /= 255
    imgs = list(image.to(device))

    model_w.eval()
    preds = model(image)

    obj = { 'pred': plot_image(imgs[0], preds)}
    
    return json.dumps(obj)
