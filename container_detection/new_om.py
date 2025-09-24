# coding=utf-8

from abc import abstractmethod, ABC  # 用于定义抽象类

import cv2 
import numpy as np 
import torch 

import acl

from det_utils import get_labels_from_txt, letterbox, scale_coords, nms  # 模型前后处理相关函数（去除draw_bbox，因为我们只需计数）

DEVICE_ID = 0 
SUCCESS = 0 
FAILED = 1 
ACL_MEM_MALLOC_NORMAL_ONLY = 2 

trained_model_path = '/home/HwHiAiUser/task5/new_model/best.om' 
image_path = '/home/HwHiAiUser/task5/img_res_with_boxes.png' 
labels_path = '/home/HwHiAiUser/task5/labels.txt' 

def init_acl(device_id):
    acl.init()
    ret = acl.rt.set_device(device_id)  # 指定运算的Device
    if ret:  
        raise RuntimeError(ret)
    context, ret = acl.rt.create_context(device_id)  # 显式创建一个Context
    if ret:
        raise RuntimeError(ret)
    print('Init ACL Successfully')
    return context


def deinit_acl(context, device_id):
    ret = acl.rt.destroy_context(context)  # 释放 Context
    if ret:
        raise RuntimeError(ret)
    ret = acl.rt.reset_device(device_id)  # 释放Device
    if ret:
        raise RuntimeError(ret)
    ret = acl.finalize()  # 去初始化
    if ret:
        raise RuntimeError(ret)
    print('Deinit ACL Successfully')


class Model(ABC):
    def __init__(self, model_path):
        print(f"load model {model_path}")
        self.model_path = model_path  # 模型路径
        self.model_id = None  # 模型 id
        self.input_dataset = None  # 输入数据结构
        self.output_dataset = None  # 输出数据结构
        self.model_desc = None  # 模型描述信息
        self._input_num = 0  # 输入数据个数
        self._output_num = 0  # 输出数据个数
        self._output_info = []  # 输出信息列表
        self._is_released = False  # 资源是否被释放
        self._init_resource()

    def _init_resource(self):
        ''' 初始化模型、输出相关资源。相关数据类型: aclmdlDesc aclDataBuffer aclmdlDataset'''
        print("Init model resource")
        # 加载模型文件
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)  # 加载模型
        self.model_desc = acl.mdl.create_desc()  # 初始化模型信息对象
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)  # 根据模型获取描述信息
        print("[Model] Model init resource stage success")

        # 创建模型输出 dataset 结构
        self._gen_output_dataset()  # 创建模型输出dataset结构

    def _gen_output_dataset(self):
        ''' 组织输出数据的dataset结构 '''
        ret = SUCCESS
        self._output_num = acl.mdl.get_num_outputs(self.model_desc)  # 获取模型输出个数
        self.output_dataset = acl.mdl.create_dataset()  # 创建输出dataset结构
        for i in range(self._output_num):
            temp_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, i)  # 获取模型输出个数
            temp_buffer, ret = acl.rt.malloc(temp_buffer_size, ACL_MEM_MALLOC_NORMAL_ONLY)  # 为每个输出申请device内存
            dataset_buffer = acl.create_data_buffer(temp_buffer, temp_buffer_size)  # 创建输出的data buffer结构,将申请的内存填入data buffer
            _, ret = acl.mdl.add_dataset_buffer(self.output_dataset, dataset_buffer)  # 将 data buffer 加入输出dataset

        if ret == FAILED:
            self._release_dataset(self.output_dataset)   # 失败时释放dataset
        print("[Model] create model output dataset success")

    def _gen_input_dataset(self, input_list):
        ''' 组织输入数据的dataset结构 '''
        ret = SUCCESS
        self._input_num = acl.mdl.get_num_inputs(self.model_desc)  # 获取模型输入个数
        self.input_dataset = acl.mdl.create_dataset()  # 创建输入dataset结构
        for i in range(self._input_num):
            item = input_list[i]  # 获取第 i 个输入数据
            data_ptr = acl.util.bytes_to_ptr(item.tobytes())  # 获取输入数据字节流
            size = item.size * item.itemsize  # 获取输入数据字节数
            dataset_buffer = acl.create_data_buffer(data_ptr, size)  # 创建输入dataset buffer结构, 填入输入数据
            _, ret = acl.mdl.add_dataset_buffer(self.input_dataset, dataset_buffer)  # 将dataset buffer加入dataset

        if ret == FAILED:
            self._release_dataset(self.input_dataset)  # 失败时释放dataset
        print("[Model] create model input dataset success")
    
    def _unpack_bytes_array(self, byte_array, shape, datatype):
        ''' 将内存不同类型的数据解码为numpy数组 '''
        np_type = None

        # 获取输出数据类型对应的numpy数组类型和解码标记
        if datatype == 0:  # ACL_FLOAT
            np_type = np.float32
        elif datatype == 1:  # ACL_FLOAT16
            np_type = np.float16
        elif datatype == 3:  # ACL_INT32
            np_type = np.int32
        elif datatype == 8:  # ACL_UINT32
            np_type = np.uint32
        else:
            print("unsurpport datatype ", datatype)
            return

        # 将解码后的数据组织为numpy数组,并设置shape和类型
        return np.frombuffer(byte_array, dtype=np_type).reshape(shape)
    
    def _output_dataset_to_numpy(self):
        ''' 将模型输出解码为numpy数组 '''
        dataset = []
        # 遍历每个输出
        for i in range(self._output_num):
            buffer = acl.mdl.get_dataset_buffer(self.output_dataset, i)  # 从输出dataset中获取buffer
            data_ptr = acl.get_data_buffer_addr(buffer)  # 获取输出数据内存地址
            size = acl.get_data_buffer_size(buffer)  # 获取输出数据字节数
            narray = acl.util.ptr_to_bytes(data_ptr, size)  # 将指针转为字节流数据

            # 根据模型输出的shape和数据类型,将内存数据解码为numpy数组
            dims = acl.mdl.get_output_dims(self.model_desc, i)[0]["dims"]  # 获取每个输出的维度
            datatype = acl.mdl.get_output_data_type(self.model_desc, i)  # 获取每个输出的数据类型
            output_nparray = self._unpack_bytes_array(narray, tuple(dims), datatype)  # 解码为numpy数组
            dataset.append(output_nparray)
        return dataset
    
    def execute(self, input_list):
        '''创建输入dataset对象, 推理完成后, 将输出数据转换为numpy格式'''
        self._gen_input_dataset(input_list)  # 创建模型输入dataset结构
        ret = acl.mdl.execute(self.model_id, self.input_dataset, self.output_dataset)  # 调用离线模型的execute推理数据
        out_numpy = self._output_dataset_to_numpy()  # 将推理输出的二进制数据流解码为numpy数组, 数组的shape和类型与模型输出规格一致
        return out_numpy

    def release(self):
        ''' 释放模型相关资源 '''
        if self._is_released:
            return

        print("Model start release...")
        self._release_dataset(self.input_dataset)  # 释放输入数据结构
        self.input_dataset = None  # 将输入数据置空
        self._release_dataset(self.output_dataset)  # 释放输出数据结构
        self.output_dataset = None  # 将输出数据置空
        
        if self.model_id:
            ret = acl.mdl.unload(self.model_id)  # 卸载模型
        if self.model_desc:
            ret = acl.mdl.destroy_desc(self.model_desc)  # 释放模型描述信息
        self._is_released = True
        print("Model release source success")

    def _release_dataset(self, dataset):
        ''' 释放 aclmdlDataset 类型数据 '''
        if not dataset:
            return
        num = acl.mdl.get_dataset_num_buffers(dataset)  # 获取数据集包含的buffer个数
        for i in range(num):
            data_buf = acl.mdl.get_dataset_buffer(dataset, i)  # 获取buffer指针
            if data_buf:
                ret = acl.destroy_data_buffer(data_buf)  # 释放buffer
        ret = acl.mdl.destroy_dataset(dataset)  # 销毁数据集

    @abstractmethod
    def infer(self, inputs):
        pass


class YoloV8(Model):
    def __init__(self, model_path):
        super().__init__(model_path)
        self.neth = 640  # 缩放的目标高度
        self.netw = 640  # 缩放的目标宽度
        self.conf_threshold = 0.2  # 置信度阈值
        self.iou_threshold = 0.5  # IoU阈值

    def infer(self, img_bgr):
        labels_dict = get_labels_from_txt(labels_path)  # 加载类别信息

        # 数据前处理
        img, scale_ratio, pad_size = letterbox(img_bgr, new_shape=[self.neth, self.netw])  # 缩放与填充
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
        img = np.ascontiguousarray(img, dtype=np.float32) / 255.0  # 转换为内存连续存储的数组
        
        # 模型推理
        output = self.execute([img, ])[0]  # 获取模型输出
        print(f"66:{output.shape}")

        # 后处理
        boxout = nms(torch.tensor(output), conf_thres=self.conf_threshold, iou_thres=self.iou_threshold)  # 非极大值抑制
        pred_all = boxout[0].numpy()  # 转换为numpy数组
        scale_coords([self.neth, self.netw], pred_all[:, :4], img_bgr.shape, ratio_pad=(scale_ratio, pad_size))  # 缩放到原始图片大小

        # 计数集装箱并绘制标记框
        container_count = 0
        img_with_boxes = img_bgr.copy()  # 复制原始图片以绘制边界框
        for det in pred_all:
            if det[4] >= self.conf_threshold and int(det[5]) == 0:  # 类ID 0 为集装箱
                container_count += 1
                # 绘制边界框和标签
                img_with_boxes = cv2.rectangle(
                    img_with_boxes, 
                    (int(det[0]), int(det[1])), 
                    (int(det[2]), int(det[3])), 
                    (0, 255, 0),  # 绿色边界框
                    2
                )
                img_with_boxes = cv2.putText(
                    img_with_boxes, 
                    f"{labels_dict[int(det[5])]} {det[4]:.2f}", 
                    (int(det[0]), int(det[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 0, 255),  # 红色文本
                    1
                )

        # 保存带有标记框的图片
        output_image_path = 'img_res_with_boxes.png'
        cv2.imwrite(output_image_path, img_with_boxes)
        print(f"带有标记框的图片已保存至: {output_image_path}")
        print(f"检测到的集装箱数量: {container_count}")

        return container_count, img_with_boxes




if __name__ == '__main__':
    context = init_acl(DEVICE_ID)  # 初始化acl相关资源
    det_model = YoloV8(model_path=trained_model_path)  # 初始化模型

    # 读入文件并推理
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # 读入图片
    count = det_model.infer(img)  # 前处理、推理、后处理, 得到集装箱数量
    print(f"Detected container count: {count}")

    # 释放相关资源
    det_model.release()  # 释放 acl 模型相关资源, 包括输入数据、输出数据、模型等
    deinit_acl(context, 0)  # acl 去初始化


