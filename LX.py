import os
import shutil
path = r'D:\aalixiang\datasets\ssaf\ssaf\glyph'

path1 = sorted([os.path.join(path, i) for i in os.listdir(path)])
cnt = 0
for i in path1:
    cnt += 1
    name = os.path.join(i, '0001.png')
    shutil.copy(name, r'C:\Users\LX\Desktop\数据集\字形/'+str(cnt).zfill(4)+'.png')