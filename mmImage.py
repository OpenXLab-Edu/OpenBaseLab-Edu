from copy import deepcopy
import cv2
import os
import numpy as np 

class MMImage:
    def __init__ (self,method='blur'):
        self.method = method
        self.img = None

    def process(self,save_path = '',para = []):
        if save_path != '':
            save_path = save_path
        img_out = getattr(self, "_"+self.method)(para)
        cv2.imwrite(save_path,img_out)

    def load_image(self,image_path):
        img = cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
        self.img = img

    def _blur(self,para = []):
        type_method,kernal = para
        img_out = cv2.blur(self.img, (kernal, kernal))  #sum(square)/25
        return img_out

    def _contour(self,para=[]):
        gray_img=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY) 
        dep,img_bin=cv2.threshold(gray_img,128,255,cv2.THRESH_BINARY) 
        image_,contours=cv2.findContours(img_bin, mode=cv2.RETR_TREE,  method=cv2.CHAIN_APPROX_SIMPLE) 
        to_write = self.img.copy() 
        # cv2.drawContours(img,contours,0,(0,0,255),3)  
        ret = cv2.drawContours(to_write,image_,-1,(0,0,255),3) 
        return ret
    
    def _hist(self,para=[]):
        gray_img=cv2.cvtColor(self.img,cv2.COLOR_BGR2GRAY) 
        img =  gray_img.astype(np.uint8)
        return cv2.equalizeHist(img)
    
    def _watershed_contour(self,para=[]):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 转为灰度图像

        # 查找和绘制图像轮廓
        Gauss = cv2.GaussianBlur(gray, (5,5), sigmaX=4.0)
        grad = cv2.Canny(Gauss,50,150)

        grad, contours = cv2.findContours(grad, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 查找图像轮廓
        markers = np.zeros(self.img.shape[:2], np.int32)  # 生成标识图像，所有轮廓区域标识为索引号 (index)
        for index in range(len(contours)):  # 用轮廓的索引号 index 标识轮廓区域
            markers = cv2.drawContours(markers, grad, index, (index, index, index), 1, 8, contours)
        ContoursMarkers = np.zeros(self.img.shape[:2], np.uint8)
        ContoursMarkers[markers>0] = 255  

        # 分水岭算法
        markers = cv2.watershed(self.img, markers)  # 所有轮廓的像素点被标注为 -1
        WatershedMarkers = cv2.convertScaleAbs(markers)
        # 用随机颜色填充分割图像
        bgrMarkers = np.zeros_like(self.img)
        for i in range(len(contours)): 
            colorKind = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
            bgrMarkers[markers==i] = colorKind
        bgrFilled = cv2.addWeighted(self.img, 0.67, bgrMarkers, 0.33, 0) 
        return cv2.cvtColor(bgrFilled, cv2.COLOR_BGR2RGB)

    def _watershed(self,para):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)  # 转为灰度图像

        # 图像的形态学梯度
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))  # 生成 5*5 结构元
        grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)  # 形态学梯度

        # 阈值分割，将灰度图像分为黑白二值图像
        _, thresh = cv2.threshold(np.uint8(grad), 0.2*grad.max(), 255, cv2.THRESH_BINARY)
        # 形态学操作，生成 "确定背景" 区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 生成 3*3 结构元
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)  # 开运算，消除噪点
        sure_bg = cv2.dilate(opening, kernel, iterations=3)  # 膨胀操作，生成 "确定背景" 区域
        # 距离变换，生成 "确定前景" 区域
        distance = cv2.distanceTransform(opening, cv2.DIST_L2, 5)  # DIST_L2: 3/5
        _, sure_fg = cv2.threshold(distance, 0.1 * distance.max(), 255, 0)  # 阈值选择 0.1*max 效果较好
        sure_fg = np.uint8(sure_fg)
        # 连通域处理
        ret, component = cv2.connectedComponents(sure_fg, connectivity=8)  # 对连通区域进行标号，序号为 0-N-1
        markers = component + 1  # OpenCV 分水岭算法设置标注从 1 开始，而连通域编从 0 开始
        kinds = markers.max()  # 标注连通域的数量
        maxKind = np.argmax(np.bincount(markers.flatten()))  # 出现最多的序号，所占面积最大，选为底色
        markersBGR = np.ones_like(self.img) * 255
        for i in range(kinds):
            if (i!=maxKind):
                colorKind = [np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)]
                markersBGR[markers==i] = colorKind
        # 去除连通域中的背景区域部分
        unknown = cv2.subtract(sure_bg, sure_fg)  # 待定区域，前景与背景的重合区域
        markers[unknown == 255] = 0  # 去掉属于背景的区域 (置零)
        # 分水岭算法标注目标的轮廓
        markers = cv2.watershed(self.img, markers)  # 分水岭算法，将所有轮廓的像素点标注为 -1
        kinds = markers.max()  # 标注连通域的数量

        # 把轮廓添加到原始图像上
        imgWatershed = self.img.copy()
        imgWatershed[markers == -1] = [0, 0, 255]  # 将分水岭算法标注的轮廓点设为红色
        # print(self.img.shape, markers.shape, markers.max(), markers.min(), ret)
        return cv2.cvtColor(markersBGR, cv2.COLOR_BGR2RGB)


    def _canny(self,para=[100,200]):
        return cv2.Canny(self.img,para[0],para[1])

    def _corner(self,para = 0.01):
        gray_img = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        gray_img = np.float32(gray_img)
        Harris_detector = cv2.cornerHarris(gray_img, 2, 3, 0.04)
        dst = cv2.dilate(Harris_detector, None)
        # 设置阈值
        thres = para*dst.max()
        img_out = deepcopy(self.img)
        img_out[dst > thres] = [255,0,0]
        return img_out
