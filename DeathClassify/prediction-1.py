"""
分割细胞图像并预测细胞周期
"""
from __future__ import print_function, unicode_literals, absolute_import, division, annotations

import csv
import json
import logging
import math
import os
from copy import deepcopy
from typing import Tuple, List

import matplotlib.pyplot as plt
import tifffile

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
import time
import cv2
import retry
import hashlib
import tensorflow as tf
from tensorflow.python.framework.errors_impl import ResourceExhaustedError
import numpy as np
from csbdeep.utils import normalize
from stardist.models import StarDist2D
from train import  get_model
import utils, config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

root_logger = logging.getLogger()
for h in root_logger.handlers:
    root_logger.removeHandler(h)

logging.basicConfig(format="%(asctime)s %(levelname)s: %(message)s", datefmt="%Y-%M-%d %H:%M:%S", level=logging.INFO)
sys.stdout = open(os.devnull, 'w')

SEG_DEV = config.SEG_DEV


class Predictor:
    """
    一个预测器，用于预测每个细胞所处的细胞周期，默认不需要提供任何参数，如需改变，请修改config.py相应的值
    """

    def __init__(self, times=config.TIMES):
        self.model = get_model()
        self.model.load_weights(filepath=config.save_model_dir_20x_best)
        # self.model.load_weights(filepath=config.save_model_dir_20x)

    def predict(self, images):
        """
        :param images: 一个包含多张图片的数组或列表，其形状为[image_count, image_width, image_height, image_channels]
        :return: 每个细胞的预测周期组成的列表
        """
        phaseMap = {0: 'D', 1: 'N'}
        img = images
        # img = cv2.resize(img, (128, 128)) / 255.0
        tensor = tf.convert_to_tensor(np.array(img))
        # tensor = tf.convert_to_tensor(img, dtype=tf.float64)

        prediction = self.model(tensor, training=False)
        # print(prediction)
        phases = []
        raw = []
        for i in prediction:
            # logging.info(i)
            phase = np.argwhere(i == np.max(i))[0][0]
            # logging.info(phase)
            # logging.info(phaseMap[phase])
            phases.append(phaseMap.get(phase))
            raw.append(phase)
        logging.info(f"raw: {raw}")
        logging.info(f"Normal: {phases.count('N')}  Death: {phases.count('D')}")
        return phases


class Prediction:
    def __init__(self, mcy: np.ndarray, dic: np.ndarray, rois, predictor=None):
        self.imgMcy = mcy
        self.imgDic = dic
        self.rois = rois
        if predictor is None:
            self.predictor = Predictor()
        else:
            self.predictor = predictor

    @staticmethod
    def __convert_dtype(__image: np.ndarray) -> np.ndarray:
        """将图像从uint16转化为uint8"""
        min_16bit = np.min(__image)
        max_16bit = np.max(__image)
        image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    # def getCell(self, cellfilter=20):
    #     """单细胞图像生成器
    #     """
    #     image_data = []
    #     for img in self.images:
    #         instances = self.parser.idMap.get(img)
    #         for instance in instances:
    #             data = utils.Data()
    #             coord = instances[instance]
    #             if type(self.imgMcy) is np.ndarray:
    #                 image_mcy = self.imgMcy
    #             else:
    #                 raise TypeError(
    #                     f"need image array, but got type {type(self.imgMcy)} in {self.imgMcy}")
    #             if type(self.imgDic) is np.ndarray:
    #                 image_dic = self.imgDic
    #             else:
    #                 raise TypeError(
    #                     f"need  image array, but got type {type(self.imgDic)} in {self.imgDic}")
    #             x0 = int(np.min(coord[0]))
    #             x1 = math.ceil(np.max(coord[0]))
    #             y0 = int(np.min(coord[1]))
    #             y1 = math.ceil(np.max(coord[1]))
    #             if np.min([(x1 - x0), (y1 - y0)]) < 5:
    #                 print(f'filter{x1 - x0, y1 - y0} ')
    #                 continue
    #             data.image_mcy = image_mcy[y0: y1, x0: x1]
    #             data.image_dic = image_dic[y0: y1, x0: x1]
    #
    #             if max(data.image_mcy.shape) < cellfilter:
    #                 print(data.image_mcy.shape)
    #                 continue
    #             cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/dic/' + instance + '.tif',
    #                         self.__convert_dtype(data.image_dic))
    #             cv2.imwrite('/home/zje/CellClassify/predict_data/LogData/mcy/' + instance + '.tif',
    #                         self.__convert_dtype(data.image_mcy))
    #             # data.image_mcy = cv2.resize(data.image_mcy, (config.image_width, config.image_height))
    #             # data.image_dic = cv2.resize(data.image_dic, (config.image_width, config.image_height))
    #             data.image_mcy = cv2.resize(self.__convert_dtype(data.image_mcy),
    #                                         (config.image_width, config.image_height))
    #             data.image_dic = cv2.resize(self.__convert_dtype(data.image_dic),
    #                                         (config.image_width, config.image_height))
    #             data.image_id = instance
    #             image_data.append(data)
    #     return image_data

    def getCell(self):
        """从分割结果中获取每个细胞的DIC和PCNA的图像信息，并返回相应的Data对象列表，
        同时会过滤掉过于小的实例，一并返回过滤后的roi信息"""
        image_data = []
        rois_after_filter = []
        index = 0
        for i in self.rois:
            x0 = int(np.min(i[0]))
            x1 = math.ceil(np.max(i[0]))
            y0 = int(np.min(i[1]))
            y1 = math.ceil(np.max(i[1]))
            w = x1 - x0
            h = y1 - y0
            # 计算增加的尺寸
            w_increase = w * 0.30
            h_increase = h * 0.30
            # 计算新的坐标点
            new_x0 = x0 - math.floor(w_increase)
            new_x1 = x1 + math.ceil(w_increase)
            new_y0 = y0 - math.floor(h_increase)
            new_y1 = y1 + math.ceil(h_increase)

            x0 = new_x0
            x1 = new_x1
            y0 = new_y0
            y1 = new_y1

            if x0 < 0:
                x0 = 0
            if y0 < 0:
                y0 = 0
            __mcy = self.imgMcy[x0:x1, y0:y1]
            __dic = self.imgDic[x0:x1, y0:y1]
            fname = f"G:\DeathDataset\debug\img-{index}.png"

            # if 0 in __mcy.shape or np.max(__mcy.shape) < 30 or np.min(__mcy.shape)<20:
            #     # print("filter instance: ", __mcy.shape)
            #     continue
            rois_after_filter.append(i)
            # mcy = self.__convert_dtype(__mcy)
            # dic = self.__convert_dtype(__dic)
            # plt.imsave(fname, mcy, cmap='gray')
            index += 1
            instance_id = hashlib.md5(str(i).encode()).hexdigest()
            data = utils.Data()
            data.image_dic = cv2.resize(__dic, (config.image_width, config.image_height)) / 255
            data.image_mcy = cv2.resize(__mcy, (config.image_width, config.image_height)) / 255
            data.image_id = instance_id
            image_data.append(data)
        return image_data, rois_after_filter

    def predict_phase(self, images: np.ndarray):
        """利用预测器预测细胞周期， 接受一个数组图像数据做输入"""
        return self.predictor.predict(images)

    def predict(self) -> Tuple[np.ndarray, List]:
        """返回单帧图像中所有细胞的预测周期， 以及过滤后的所有细胞轮廓信息"""
        image_data = []
        id_data = []
        cells, rois_after_filter = self.getCell()
        for cell in cells:
            data = np.dstack([cell.image_dic, cell.image_mcy])
            image_data.append(data)
            id_data.append(cell.image_id)
        images = np.array(image_data)
        phases = self.predict_phase(images)
        return phases, rois_after_filter

    def exportResult(self):
        pass


class Segmenter(object):
    def __init__(self, segment_model=None):
        if segment_model is None:
            sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            if SEG_DEV:
                self.model = StarDist2D(None, name=config.segment_dev_model_name,
                                   basedir=config.segment_dev_model_basedir)
            else:
                self.model = StarDist2D(None, name=config.segment_model_name_20x,
                                    basedir=config.segment_model_saved_dir_20x)
        else:
            self.model = segment_model

    @retry.retry(exceptions=ResourceExhaustedError)
    def segment(self, image: np.ndarray):
        img = np.dstack([image, image])
        axis_norm = (0, 1)  # normalize channels independently
        if SEG_DEV:
            im = normalize(img, 1, 99.8, axis=axis_norm)
        else:
            im = normalize(image, 1, 99.8, axis=axis_norm)
        labels, details = self.model.predict_instances(im)
        return labels, details

    def segment_v2(self, image_dic, image_mcy):
        axis_norm = (0, 1)
        img = np.dstack([ normalize(image_mcy, 1, 99.8, axis=axis_norm),  normalize(image_dic, 1, 99.8, axis=axis_norm)])
        # im = normalize(img, 1, 99.8, axis=axis_norm)
        labels, details = self.model.predict_instances(img)
        return labels, details

class Segmentation(object):
    def __init__(self,
                 image_mcy: np.ndarray,
                 image_dic: np.ndarray,
                 imagename: str,
                 segmenter=None,
                 predictor=None,
                 roi_save_path=None,
                 label_save_path=None):
        self.imageName = imagename.replace('.tif', '.png')
        if segmenter is None:
            self.segmentor = Segmenter()
        else:
            self.segmentor = segmenter
        self.predictor = predictor
        self.roi_save_path = roi_save_path
        self.label_save_path = label_save_path
        self.img_mcy = image_mcy
        self.img_dic = image_dic

    def segment(self):
        if SEG_DEV:
            return self.segmentor.segment_v2(self.img_dic, self.img_mcy)
        return self.segmentor.segment(image=self.img_mcy)

    @property
    def labels(self):
        return self.segment()[0]

    @property
    def details(self):
        return self.segment()[1]

    def rois(self):
        return self.details['coord']

    def class_info(self):
        return self.details['class_id']

    def save_labels(self, label_save_path=None):
        from csbdeep.io import save_tiff_imagej_compatible
        if label_save_path:
            save_tiff_imagej_compatible(label_save_path, self.labels, axes='YX')
        else:
            save_tiff_imagej_compatible(self.label_save_path, self.labels, axes='YX')

    def save_rois(self, roi_save_path=None):
        from stardist import export_imagej_rois
        if roi_save_path:
            export_imagej_rois(roi_save_path, self.rois())
        elif self.roi_save_path:
            export_imagej_rois(self.roi_save_path, self.rois())
        else:
            raise ValueError('roi_save_path need to be give, in this function or in the class init')

    def __get_segment_result(self):
        """将分割的坐标转化为可识读的json字典"""
        rois = self.rois()
        # c = ConverterXY(self.imageName.replace('.tif', '.png'), coordinates=rois)
        # return c.json
        return rois

    @staticmethod
    def __convert_dtype(__image):
        min_16bit = np.min(__image)
        max_16bit = np.max(__image)
        image_8bit = np.array(np.rint(255 * ((__image - min_16bit) / (max_16bit - min_16bit))), dtype=np.uint8)
        return image_8bit

    def _predict(self):
        """just a test function, do not use or call it!!!"""
        images = []
        coords = self.rois()
        for i in coords:
            x0 = int(np.min(i[0]))
            x1 = math.ceil(np.max(i[0]))
            y0 = int(np.min(i[1]))
            y1 = math.ceil(np.max(i[1]))
            test_mcy = self.img_mcy[x0:x1, y0:y1]
            test_dic = self.img_dic[x0:x1, y0:y1]
            if 0 in test_mcy.shape:
                continue
            test_mcy = self.__convert_dtype(test_mcy)
            test_dic = self.__convert_dtype(test_dic)
            test_img = np.dstack([cv2.resize(test_dic, (100, 100)), cv2.resize(test_mcy, (100, 100))])
            images.append(test_img / 255)
            # print(Predictor().predict(np.array([test_img])))
        ret = Predictor().predict(np.array(images))


    def __add_predict_phase(self):
        rois = self.__get_segment_result()
        regions = []
        regions_tmp = {
            "shape_attributes":
                {
                    "name": "polygon",
                    "all_points_x": [],
                    "all_points_y": []
                },
            "region_attributes":
                {
                    "phase": None,
                }
        }

        predictor = Prediction(mcy=self.img_mcy, dic=self.img_dic, rois=rois,
                               predictor=self.predictor)

        phases, rois_after_filter = predictor.predict()
        assert len(rois_after_filter) == len(phases)
        for i in zip(rois, phases):
            all_x = []
            all_y = []
            for j in range(i[0].shape[1]):
                all_x.append(float(i[0][:, j][1]))
                all_y.append(float(i[0][:, j][0]))
            death = i[1]
            t = deepcopy(regions_tmp)
            t["shape_attributes"]["all_points_x"] = all_x
            t["shape_attributes"]["all_points_y"] = all_y
            t["region_attributes"]["phase"] = death
            regions.append(t)
        tmp = {
            self.imageName: {
                "filename": self.imageName,
                "size": 4194304,
                "regions": regions,
                "file_attributes": {}
            }
        }
        pl = list(phases)
        info = (len(pl),  pl.count('D'), pl.count('N'))
        return tmp, info

    @property
    def predict_result(self):
        return self.__add_predict_phase()

    def export_predict_result(self, filepath):
        """导出分割和预测细胞周期到json文件中"""
        with open(filepath, 'w') as f:
            json.dump(self.__add_predict_phase(), f)


def segment(pcna: os.PathLike | str, bf: os.PathLike | str, output: os.PathLike | str, segment_model=None, xrange=None):
    jsons = {}
    mcy_data = utils.readTif(filepath=pcna)
    dic_data = utils.readTif(filepath=bf)
    segmenter = Segmenter(segment_model=segment_model)
    predictor = Predictor()
    _xrange = xrange
    record_info = []
    while True:
        try:
            # mcy_img, imagename = next(mcy_data)
            mcy_img, imagename = next(mcy_data)
            dic_img, _ = next(dic_data)
            # mcy_img = utils.divideImage(mcy_img, 4, 4)[0]
            # dic_img = utils.divideImage(dic_img, 4, 4)[0]
            start_time = time.time()
            seg = Segmentation(image_mcy=mcy_img,
                               imagename=imagename,
                               image_dic=dic_img,
                               segmenter=segmenter,
                               predictor=predictor)
            value, predict_phase_info = seg.predict_result
            jsons.update(value)
            end_time = time.time()
            logging.info(f'segment {os.path.basename(imagename)}; cost time: {end_time - start_time:.2f}s')
            logging.info(predict_phase_info)
            record_info.append((os.path.basename(imagename), *predict_phase_info))
            del seg
            if _xrange is not None:
                _xrange -= 1
                if not _xrange:
                    break
        except StopIteration:
            break
    statistic_filename = pcna.replace('.tif', '.csv')
    with open(statistic_filename, 'w', newline='') as fc:
        writer = csv.writer(fc)
        writer.writerow(['frame_name', 'all_count', 'death_count', 'normal_count'])
        writer.writerows(record_info)
    json_filename = os.path.basename(pcna).replace('.tif', '.json')
    if output:
        out = output
    else:
        out = json_filename
    with open(out, 'w') as f:
        json.dump(jsons, f)


    return jsons


def run(pcna, bf, output):
    segment(pcna, bf, output)
    mcy_data = tifffile.imread(pcna)[0,:,:]
    dic_data = tifffile.imread(bf)[0,:,:]
    segmenter = Segmenter(segment_model=None)
    predictor = Predictor()
    mcy_img = mcy_data
    dic_img = dic_data
    imagename = 'mcy.tif'
    # mcy_img = utils.divideImage(mcy_img, 4, 4)[0]
    # dic_img = utils.divideImage(dic_img, 4, 4)[0]
    start_time = time.time()
    jsons = {}
    seg = Segmentation(image_mcy=mcy_img,
                       imagename=imagename,
                       image_dic=dic_img,
                       segmenter=segmenter,
                       predictor=predictor)
    value, predict_phase_info = seg.predict_result
    jsons.update(value)
    end_time = time.time()
    logging.info(f'segment {os.path.basename(imagename)}; cost time: {end_time - start_time:.2f}s')
    logging.info(predict_phase_info)
    del seg
    json_filename = os.path.basename(pcna).replace('.tif', '.json')
    if output:
        out = output
    else:
        out = json_filename
    with open(out, 'w') as f:
        json.dump(jsons, f)
    return jsons


if __name__ == '__main__':
    run(bf=r"G:\WJQ\20240101rpe-pcna-y530f+dox-inhibitor\0\copy_of_r-src-y530f-1-5L-6-10DL-11-15DZ-16-20Z-21-25NC-26-30DNC-31-37DF_0_dic.tif",
         pcna=r"G:\WJQ\20240101rpe-pcna-y530f+dox-inhibitor\0\copy_of_r-src-y530f-1-5L-6-10DL-11-15DZ-16-20Z-21-25NC-26-30DNC-31-37DF_0_pcna.tif",
         output=r'G:\WJQ\20240101rpe-pcna-y530f+dox-inhibitor\0\demo.json')




