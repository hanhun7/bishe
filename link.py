import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from net import DFMNet
from data import test_dataset
import time
# 导入Flask类
from flask import Flask
from flask import request
from flask_cors import CORS
import io
from flask import send_file
import PIL
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=256, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='./dataset/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id == '3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')
elif opt.gpu_id=='all':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    print('USE GPU 0,1,2,3')

#load the model
model = DFMNet()
model.load_state_dict(torch.load('./pretrain/DFMNet_epoch_300.pth'))
model.to(device)
model.eval()

def save(save_path,name,res,gt,notation=None,sigmoid=True):
    res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze() if sigmoid ==True else res.data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    print('save img to: ', os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name))
    cv2.imwrite(os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name), res * 255)
    return os.path.join(save_path, name.replace('.png','_'+notation+'.png') if notation != None else name)

# 创建Flask实例，接收参数__name__表示当前文件名，默认会自动指定静态文件static
app = Flask(__name__)
# 跨域注释
CORS(app,resources=r'/*')
# 装饰器的作用是将路由映射到视图函数get_predict；告诉flask通过哪个URL可以触发函数
@app.route('/get_predict',methods=['POST'])
def get_age():
    # 接收传来的图片
    img_AcceptR = request.files.get('fileR')
    img_AcceptD = request.files.get('fileD')
    print(img_AcceptR.filename)
    print(img_AcceptD.filename)

    # 设置图片要保存到的路径
    pathR = 'D:/毕业设计/算法/imgR/'
    pathD = 'D:/毕业设计/算法/imgD/'
    loca = time.strftime('%Y%m%d%H%M%S')
    newFileR=pathR+loca
    newFileD=pathD+loca
    os.makedirs(newFileR)
    os.makedirs(newFileD)
    # 获取图片名称及后缀名
    imgNameR = img_AcceptR.filename
    imgNameD = img_AcceptD.filename
    # 图片path和名称组成图片的保存路径
    file_pathR = newFileR +'/'+ imgNameR
    file_pathD = newFileD +'/'+ imgNameD
    # 保存图片
    img_AcceptR.save(file_pathR)
    img_AcceptD.save(file_pathD)
    save_path = 'D:/imgRe/'

    with torch.no_grad():
        print('111')
        image_root = newFileR+'/'
        gt_root = './GT/'
        depth_root = newFileD+'/'
        test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
        url='D:/imgRe/20220307210436983.png'
        print(test_loader.size)
        for i in range(test_loader.size):
            image, gt, depth, name, image_for_post = test_loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.to(device)
            depth = depth.to(device)
            # torch.cuda().synchronize()
            time_s = time.time()
            out = model(image, depth)
            print(out)
            # torch.cdua().synchronize()
            time_e = time.time()
            t = time_e - time_s
            print("time: {:.2f} ms".format(t * 1000))
            url=save(save_path,name,out[0],gt)

    print('Test Done!')


    with open(url, 'rb') as f:
        a = f.read()

    return send_file(
        io.BytesIO(a),
        mimetype='image/png',
        as_attachment=True,
        download_name='result.png'
    )

# Flask应用程序实例的run方法启动WEB服务器
if __name__ == '__main__':
    app.run(host='127.0.0.1',port=5000)  # 127.0.0.1 #指的是本地ip