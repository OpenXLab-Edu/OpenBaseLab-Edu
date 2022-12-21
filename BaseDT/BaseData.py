# 导入工具库
import numpy as np
import cv2
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer

class SingleData(object):
    '''
    目前支持图片路径, np array, 文本
    后续支持 PIL
    '''
    _defaults = {
        "data_type": 'uint8',
        # 图像默认属性
        "to_rgb": True,
        # 文本默认属性
        "vct": {"str": CountVectorizer(), "dict": DictVectorizer()},
    }

    def get_attribute(self, n):
        if n in self.__dict__:
            return self.__dict__[n]
        else:
            return None

    def __init__(self, data_source, data_class = 'img',**kwargs):
        self.vct = None
        self.data_source = data_source
        self.data_class = data_class
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)

        if self.data_class == "img":
            if type(self.data_source) == str:
                self.value = cv2.imdecode(np.fromfile((self.data_source),dtype=np.uint8),-1)
            if type(self.data_source) == np.ndarray:
                self.value = self.data_source
            else:
                # TODO 检查合法性
                pass

            self.value = self.value.astype(self.get_attribute("data_type"))
            self.raw_value = self.value
            if self.get_attribute("to_rgb") == False:
                self._rgb2gray()
            if self.get_attribute("size"):
                size = self.get_attribute("size")
                self._resize(size)
            if self.get_attribute("crop_size"):
                crop_size = self.get_attribute("crop_size")
                self._crop(crop_size)
            if self.get_attribute("normalize")==True:
                #TODO 需检查维度
                mean = self.get_attribute("mean")
                std = self.get_attribute("std")
                self._normalize(mean, std)

        elif self.data_class == "txt":
            #这里的data_source不可以是文件
            self.raw_value = self.data_source
            self.value = self.raw_value
            if type(self.raw_value) == dict or type(self.raw_value[0]) == dict:
                self.text_type = "dict"
                if type(self.raw_value) == dict: self.value = [self.raw_value]
            if type(self.raw_value) == str or type(self.raw_value[0]) == str:
                self.text_type = "str"
                if type(self.raw_value) == str: self.value = [self.raw_value]
            else:
                #TODO 检查合法性
                pass
            if self.get_attribute("vectorize") == True:
                self.value = self.vectorizer(self.get_attribute("chinese"), self.get_attribute("fitted"))
            if self.get_attribute("normalize")==True:
                #TODO 需检查维度
                mean = self.get_attribute("mean")
                std = self.get_attribute("std")
                self._normalize(mean, std)
        else:
            print("unsupported data class")

    def to_tensor(self):
        if self.data_class == "img" and self.get_attribute("to_rgb"):
            return np.expand_dims(np.transpose(self.value, (2,0,1)), 0)
        else:
            return np.expand_dims(np.expand_dims(self.value, 0), 0)


    #保护方法，不给用户调用
    def _rgb2gray(self):
        gray = np.dot(self.value, [0.2989, 0.5870, 0.1140])
        gray = np.clip(gray, 0, 255).astype(np.uint8)
        self.value = gray

    def _normalize(self, mean, std):
        # assert self.value.dtype != np.uint8
        img = self.value
        mean = np.float64(np.array(mean).reshape(1, -1))
        stdinv = 1 / np.float64(np.array(std).reshape(1, -1))
        if self.get_attribute("data_class") == 'img' and self.get_attribute("to_rgb"):
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
        img = img.astype(np.float32)
        cv2.subtract(img, mean, img)  # inplace
        cv2.multiply(img, stdinv, img)  # inplace
        self.value = img

    def _resize(self, size):
        #TODO 支持padding
        self.value = cv2.resize(self.value, size, interpolation=cv2.INTER_LINEAR)
        #self.value = np.resize(self.value, size, interp='bilinear')

    def _bbox_clip(self, bboxes, img_shape):
        assert bboxes.shape[-1] % 4 == 0
        cmin = np.empty(bboxes.shape[-1], dtype=bboxes.dtype)
        cmin[0::2] = img_shape[1] - 1
        cmin[1::2] = img_shape[0] - 1
        clipped_bboxes = np.maximum(np.minimum(bboxes, cmin), 0)
        return clipped_bboxes

    def _bbox_scaling(self, bboxes, scale, clip_shape=None):
        if float(scale) == 1.0:
            scaled_bboxes = bboxes.copy()
        else:
            w = bboxes[..., 2] - bboxes[..., 0] + 1
            h = bboxes[..., 3] - bboxes[..., 1] + 1
            dw = (w * (scale - 1)) * 0.5
            dh = (h * (scale - 1)) * 0.5
            scaled_bboxes = bboxes + np.stack((-dw, -dh, dw, dh), axis=-1)
        if clip_shape is not None:
            return self._bbox_clip(scaled_bboxes, clip_shape)
        else:
            return scaled_bboxes

    def _crop(self, size, scale=1.0, pad_fill=None):
        if isinstance(size, int):
            crop_size = (size, size)
        else:
            crop_size = size

        img = self.value
        img_height, img_width = img.shape[:2]

        crop_height, crop_width = crop_size

        if crop_height > img_height or crop_width > img_width:
            #TODO 可选择pad_mod
            pass
        else:
            crop_height = min(crop_height, img_height)
            crop_width = min(crop_width, img_width)

        y1 = max(0, int(round((img_height - crop_height) / 2.)))
        x1 = max(0, int(round((img_width - crop_width) / 2.)))
        y2 = min(img_height, y1 + crop_height) - 1
        x2 = min(img_width, x1 + crop_width) - 1
        bboxes = np.array([x1, y1, x2, y2])

        chn = 1 if img.ndim == 2 else img.shape[2]
        if pad_fill is not None:
            if isinstance(pad_fill, (int, float)):
                pad_fill = [pad_fill for _ in range(chn)]
            assert len(pad_fill) == chn

        _bboxes = bboxes[None, ...] if bboxes.ndim == 1 else bboxes
        scaled_bboxes = self._bbox_scaling(_bboxes, scale).astype(np.int32)
        clipped_bbox = self._bbox_clip(scaled_bboxes, img.shape)

        patches = []
        for i in range(clipped_bbox.shape[0]):
            x1, y1, x2, y2 = tuple(clipped_bbox[i, :])
            if pad_fill is None:
                patch = img[y1:y2 + 1, x1:x2 + 1, ...]
            else:
                _x1, _y1, _x2, _y2 = tuple(scaled_bboxes[i, :])
                if chn == 1:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1)
                else:
                    patch_shape = (_y2 - _y1 + 1, _x2 - _x1 + 1, chn)
                patch = np.array(
                    pad_fill, dtype=img.dtype) * np.ones(
                    patch_shape, dtype=img.dtype)
                x_start = 0 if _x1 >= 0 else -_x1
                y_start = 0 if _y1 >= 0 else -_y1
                w = x2 - x1 + 1
                h = y2 - y1 + 1
                patch[y_start:y_start + h, x_start:x_start + w,
                ...] = img[y1:y1 + h, x1:x1 + w, ...]
            patches.append(patch)

        if bboxes.ndim == 1:
            self.value = patches[0]
        else:
            self.value = patches

    def _pad(self):
        pass

    def _flip(self):
        pass

    def vectorizer(self, chinese = False, fitted = False):
        '''
        支持单文本及多文本
        :param text: 允许raw txt输入以及半处理数据例如字典输入
        :return: 词向量矩阵（词向量组成的矩阵）
        '''
        txt_cnt = None
        text_type = self.text_type
        texts = self.value

        if chinese:
            for i in range(len(texts)):
                texts[i] = self._chinese_cut(texts[i])
        #过滤
        # for i in range(len(texts)):
        #     texts[i] = re.sub('[^\u4e00-\u9fa5a-zA-Z0-9\sP]+', '', texts[i])
        if text_type == "dict":
            if not fitted: txt_cnt = self.vct[text_type].fit_transform(texts)
            if fitted: txt_cnt = self.vct[self.text_type].transform(texts)

        if text_type=="str":
            if not fitted: txt_cnt = self.vct[text_type].fit_transform(texts)
            if fitted: txt_cnt = self.vct[text_type].transform(texts)
        return txt_cnt.A

    def fit(self, texts = ""):
        '''
        生成词汇表，支持单文本及多文本
        :param text:
        :return:
        '''
        text_type = self.text_type
        self.vct[text_type] = self.vct[text_type].fit(texts)

    def get_feature_names(self):
        if self.get_attribute("data_class") != "txt":
            print("该数据类型不支持get_feature_names")
        else:
            texts = self.value
            text_type = self.text_type
            return self.vct[text_type].get_feature_names_out()

    def _chinese_cut(self, text):
        '''
        单文本操作
        :param text:
        :return: 返回字符串
        '''
        words = list(jieba.cut(text))
        text = "/ ".join(words)
        # text = "".join(words)
        return text



if __name__ == "__main__":
    img = cv2.imread("D:\PythonProject\OpenDataLab-Edu\dataset\cat2.jpg")
    data = SingleData(img,
                      to_rgb=True,
                      size=(256, 256),
                      crop_size=224,
                      normalize=True,
                      mean=[123.675, 116.28, 103.53],
                      std=[58.395, 57.12, 57.375]
                      )
    print(data.value)
    tensor_value = data.to_tensor()
    print(tensor_value)
    # texts = {'city': 'Dubai', 'temperature': 33},
    # data = SingleData(texts, data_class= "txt", vectorize = True, normalize = True, mean=[0], std=[1])
    # print(data.to_tensor())
