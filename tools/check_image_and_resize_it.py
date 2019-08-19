import cv2
import os
import numpy as np

image_dir = './images'
image_name = 'traffics-detect-all-x.bmp'
image_path = os.path.join(image_dir, image_name)
images_size = (1059, 771)    #448 336

assert os.path.isfile(image_path)
img = cv2.imread(image_path, 1)

imgsize = ' x '.join([str(s) for s in img.shape])
print('\ncurrent size is %s.' % imgsize)

img = cv2.resize(img, dsize=images_size)

fmt = '.' + image_name.split('.')[-1]
image_name = image_name.split(fmt)[0] \
             + '-resized-%s'%images_size[0] \
             + '.bmp'
             # + fmt

# 添加文字
if 1:
    label = ''
    # insize = image_name.split('--')[1].split('-')[1]
    # score = image_name.split('-top')[1].split('-')[0]
    # text = '%.2f, %s' % (round(float(score), 2), label)
    # text = '%s, %s, %s' % (insize, score[:4], label)
    text = '%s %s' % ('128x128', label)
    pos = (1+230, 10+0)
    font_type = 4
    font_size = 0.3
    color = (200, 200, 200)
    bold = 1

    # 图片，文字，位置，字体，字号，颜色，厚度
    cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_size, color, bold, cv2.LINE_AA)
    # cv2.imshow('img', img)

savepath = os.path.join(image_dir, image_name)

cv2.imwrite(savepath, img)

print('\nimage has been saved at %s' % savepath)
