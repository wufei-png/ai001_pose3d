1.python3.8 环境

2.安装paddle

pip install paddlepaddle

3.安装paddlehub

pip install paddlehub

pip install docx2pdf

conda install pyclipper

pip install pywin32

conda install shapely 

4.下载模型

hub install chinese_ocr_db_crnn_mobile

pip install python-docx

使用：
from ocr import predict
#输入要识别的图片名称
predict('test.jpg')

test.docx 文件不要删除！！