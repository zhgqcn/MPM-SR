
import PIL.Image as Image
import os

IMAGES_PATH = r'D:\\GraduationProjectBackUp\\TestImg\\result\\X4\\patch\\'
IMAGES_FORMAT = ['.jpg', '.tif'] 
IMAGE_SIZE = 512
IMAGE_ROW = 4 
IMAGE_COLUMN = 4
IMAGE_SAVE_PATH = r'D:\\GraduationProjectBackUp\\TestImg\\result\\X4\\merged1.tif'


image_names = [name for name in os.listdir(IMAGES_PATH) for item in IMAGES_FORMAT if
               os.path.splitext(name)[1] == item]

print(image_names)
if len(image_names) != IMAGE_ROW * IMAGE_COLUMN:
    raise ValueError("合成图片的参数和要求的数量不能匹配！")



def image_compose():
    to_image = Image.new('RGB', (IMAGE_COLUMN * IMAGE_SIZE, IMAGE_ROW * IMAGE_SIZE)) 
   
    i = 1
    for x in range(1, IMAGE_COLUMN + 1):
       for y in range(1, IMAGE_ROW + 1):
            from_image = Image.open(IMAGES_PATH + str(i) + '.tif').resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * IMAGE_SIZE, (y - 1) * IMAGE_SIZE))
            i = i + 1
    return to_image.save(IMAGE_SAVE_PATH)  


image_compose()  