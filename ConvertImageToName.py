import os

ImagePath = './Image'
temp = os.listdir(ImagePath)
total = []
for name in temp:
    if name.endswith('.png'):
        total.append(name)

num = len(total)

with open('ImageName.txt', 'w') as f:
    for i in range(num):
        name = total[i][:-4] + '\n'
        f.write(name)
