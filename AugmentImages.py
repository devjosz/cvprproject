import cv2 as cv
import os
import sys

IMAGE_FILE_EXTENSIONS = [".jpg", ".JPG", ".png", ".PNG"]

LR_FLIP_PREFIX = "aug_lrflip_"

def is_image(input_file):
    for extension in IMAGE_FILE_EXTENSIONS:
        if extension in input_file:
            return True
    return False

def lr_flip(input_file):
        # Load the image
        img = cv.imread(input_file)
        if img is None:
            sys.exit("Could not read image")
        # Flip (flipcode > 1 to flip left to right)
        flipped_img = cv.flip(img, 1)
        return flipped_img
    
def is_unaugmented_image(input_file):
    if ( is_image(input_file) ):
        if (not (input_file[0:len(LR_FLIP_PREFIX)] == LR_FLIP_PREFIX)):
            # Checks if the image filename has the "aug_lrflip_" prefix,
            # meaning that it has already been augmented by flipping. 
            return True
    return False

def save_aug_image(input_image, dir_name, filename, prefix):
    """
    Saves the image with the given filename (e.g. "image.jpg") located
    in the directory dir_name (so the file is located at
    dir_name/image.jpg in the example) AS "dir_name/prefiximage.jpg"
    """
    img_filename = prefix + filename
    cv.imwrite(dir_name + "/" + img_filename, input_image)
    
# for root, dirs, files in os.walk('data/train'):
#     print(root)
#     print("\n")
#     print(dirs)
#     print("\n")
#     print(files)
#     print("\n\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

train_directory = "data/train"

# for root, dirs, files in os.walk(train_directory):
#     for loc_file in files:
#         print(loc_file)
#         if is_unaugmented_image(loc_file):
#             lr_flip(loc_file)
#         else:
#             continue

for root, dirs, files in os.walk(train_directory):
    for filename in files:
        img_filename = root + "/" + filename
        #print(img_filename)
        if (is_unaugmented_image(filename)):
            flipped_img = lr_flip(img_filename)
            save_aug_image(flipped_img, root, filename, LR_FLIP_PREFIX)

print("Done!")
