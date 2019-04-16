from PIL import Image
import numpy as np

# 设定取代所使用的字符列表
char_list = list("$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\|()1{}[]?-_+~<>i!lI;:,\"^`'. ")

# 根据灰度大小决定使用哪种字符进行取代
def get_char(r,g,b,a=256):
    if a == 0:
        return ""
    gray = int(0.2126*r + 0.7152*g + 0.0722*b)
    unit = 257/len(char_list)
    return char_list[int(gray/unit)]

# 图象处理

def get_pic(path):
    img = Image.open(path)
    height = int(np.asarray(img).shape[0]*0.5)
    width = int(np.asarray(img).shape[1]*0.5)
    # 调整原图大小
    # img = img.resize((100, 75), Image.NEAREST)
    # 创建空图像字符串
    text = ""
    # 遍历图象每一个像素
    for i in range(height):
        for j in range(width):
            pixel = img.getpixel((j, i))
            text += get_char(*pixel)
        text += '\n'

    with open('out.txt', 'w') as f:
        f.write(text)
    pass

if __name__ == '__main__':
    get_pic('./image/demo.jpg')
