#!/usr/bin/python3

# python2 png2json.py test /home/wu/samplecodes/Mask_RCNN/data/my_coco

import os
import io
from PIL import Image
import numpy as np

def png2darklabel(labelsfolder='labels', datasfolder="train2014_small"):
    
    rootfolder = "/media/joey/DATA/parcel_blur_wuhe"
    labelsfolder = '%s/labels/%s' % (rootfolder, datasfolder)
    pngFolder = '%s/images/%s' % (rootfolder, datasfolder)
    print("reading from %s, writing lable to %s" %  (pngFolder, labelsfolder))
    
    # Get images in png folder
    
    Filename = os.listdir(pngFolder)

    maskNames = [fileName[:-4] for fileName in Filename 
                if (fileName.endswith('.png') 
                    and (fileName.find('mask') != -1))]
    print(maskNames)
    
    maskNames.sort()
    
    maskCount = len(maskNames)
    
    image_dict_list = []
    mask_dict_list = []
    
    # generate image txt file
    for i, maskName in zip(range(0, maskCount), maskNames):
        # todo: split imgName to get image id
        tokens = maskName.split('_')
        image = Image.open("%s/%s.png" % (pngFolder, maskName))
        #print(image.format, image.size, image.mode)
        mask_width = image.size[0]
        mask_height = image.size[1]
        #print("w, h: %s, %s" %(mask_width, mask_height))
        bbox = image.getbbox()
        print(bbox)
        labelPath = '%s/%s_%s.txt' % (labelsfolder, tokens[0],tokens[1])
        print("writing label: %s" % labelPath)
        # bbox center x, bbox center y,  bbox width, bbox height normalized with image size
        label_str = '%s %s %s %s %s\n' % (int(tokens[3]) - 1,
                                           (bbox[0] + (bbox[2] - bbox[0]) / 2) / mask_width, 
                                           (bbox[1] + (bbox[3] - bbox[1]) / 2) / mask_height, 
                                          (bbox[2] - bbox[0]) / mask_width, 
                                          (bbox[3] - bbox[1]) / mask_height)
        with io.open(labelPath, 'a', encoding = 'utf8') as output:
            output.write(label_str)

    return
        
if __name__ == '__main__':
    
    import argparse
    parser = argparse.ArgumentParser(
        description = 'Generate darknet labels from mask pngs')
    parser.add_argument('labelsfolder',
                        default='labels',
                        metavar="<labelsfolder>",
                        help="specify generated json types")
    parser.add_argument('datasetfolder',
                        default='train2014',
                        metavar="<datasetfolder>",
                        help="specify dataset folder")
        
    args = parser.parse_args()
    labelsfolder = args.labelsfolder
    datafolder = args.datasetfolder
    png2darklabel(labelsfolder, datafolder)
    