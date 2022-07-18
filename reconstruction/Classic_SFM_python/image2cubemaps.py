# from descriptors import siftAID
import sys
import cv2
import py360convert as pc
import numpy as np

# Cubemap resolution, a higher number might arise better (but slower) results
CUBEMAP_RESOLUTION = 2048

def extract_cubemaps(output_prefix, img):
    # Read images from files
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2GRAY)
    # Create cubemap faces array
    cubemaps = pc.e2c(np.expand_dims(img, axis=2), face_w=CUBEMAP_RESOLUTION, cube_format='list')[:4]
    # Flip second and third cubemap faces
    for k in [1,2]:
        for cubemap in cubemaps:
            cubemap[k] = np.flip(cubemap[k], axis=1)

    for idx,cubemap in enumerate(cubemaps):
        cv2.imwrite('%s_%d.png' % (output_prefix, idx), cubemap)

if __name__ == "__main__":
    extract_cubemaps(sys.argv[1], sys.argv[2])
