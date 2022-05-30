import linecache

import numpy as np
import pandas as pd
from PIL import Image

from siamese import Siamese


                                                        #预测不同的模型就修改相应的siamese中的路径和模型


if __name__ == "__main__":
    model = Siamese()
    txt_root = 'pre.txt'
    imagepath = "datasets/images"
    lines = linecache.getlines(txt_root)
    img0_list = []
    img1_list = []
    for line in lines:
        line.strip('\n')
        line = line.split('\t')
        img0_list.append(line[0])
        img1_list.append(line[1].strip('\n'))
    sim = []
        

    for i in range(len(img0_list)):
        img0 = Image.open(imagepath1 + '/' + img0_list[i])
        img1 = Image.open(imagepath1 + '/' + img1_list[i])
        probability = model.detect_image(img0, img1)
        print(probability)
        print(i)
        sim.append(float(probability))
    df = pd.DataFrame(sim, columns=['prediction'])
    df.to_excel("pre_list.xlsx", index=False)

