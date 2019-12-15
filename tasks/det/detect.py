import time
from data_loader.utility.images_loader import ImagesLoader
from data_loader.utility.video_loader import VideoLoader
from utility.utils import *
from torch_utility.torch_model_process import TorchModelProcess
from drawing.imageDraw import ImageDraw
from base_algorithm.non_max_suppression import NonMaxSuppression
from config import detectConfig
from tasks.det.detect_result_process import DetectResultProcess


class Detection():

    def __init__(self, cfg_path, gpu_id):
        self.imageDraw = ImageDraw()
        self.torchModelProcess = TorchModelProcess()
        self.result_process = DetectResultProcess()
        self.nms_process = NonMaxSuppression()

        self.model = self.torchModelProcess.initModel(cfg_path, gpu_id)
        self.device = self.torchModelProcess.getDevice()

    def load_weights(self, weights_path):
        self.torchModelProcess.loadLatestModelWeight(weights_path, self.model)
        self.torchModelProcess.modelTestInit(self.model)

    def detect(self, input_path):
        if os.path.isdir(input_path):
            dataloader = ImagesLoader(input_path, detectConfig.imgSize)
        else:
            dataloader = VideoLoader(input_path, detectConfig.imgSize)

        for i, (oriImg, img) in enumerate(dataloader):
            print('%g/%g' % (i + 1, len(dataloader)), end=' ')
            prev_time = time.time()
            # Get detections
            # Get detections
            with torch.no_grad():
                output_list = self.model(img.to(self.device))
                output = self.compute_output(output_list)
                result = self.result_process.get_detection_result(output, 5e-3)
            detection_objects = self.nms_process.multi_class_nms(result, detectConfig.nmsThresh)
            detection_objects = self.result_process.resize_detection_objects(oriImg,
                                                                             detectConfig.imgSize,
                                                                             detection_objects,
                                                                             detectConfig.className)
            print('Batch %d... Done. (%.3fs)' % (i, time.time() - prev_time))
            self.imageDraw.drawDetectObjects(oriImg, detection_objects)

            cv2.namedWindow("image", 0)
            cv2.resizeWindow("image", int(oriImg.shape[1] * 0.8), int(oriImg.shape[0] * 0.8))
            cv2.imshow("image", oriImg)
            if cv2.waitKey() & 0xff == ord('q'):  # 按q退出
                break

    def compute_output(self, output_list):
        count = len(output_list)
        preds = []
        for i in range(0, count):
            temp = self.model.lossList[i](output_list[i])
            preds.append(temp)
        prediction = torch.cat(preds, 1)
        prediction = prediction.squeeze(0)
        return prediction



