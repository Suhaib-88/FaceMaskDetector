import cv2
from pytorch_infer import run_on_video
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.pytorch_loader import load_pytorch_model, pytorch_inference
import time
import numpy as np
from PIL import Image

        
class Video(object):
    def __init__(self):
        pass
    def get_vids(self,cam):
                
        model = load_pytorch_model('models/model360.pth')
            # anchor configuration
            #feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
        feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
        anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
        anchor_ratios = [[1, 0.62, 0.42]] * 5
        #   generate anchors
        anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)
        # for inference , the batch size is 1, the model output shape is [1, N, 4],
        # so we expand dim for anchors to [1, anchor_num, 4]
        anchors_exp = np.expand_dims(anchors, axis=0)

        id2class = {0: 'Mask', 1: 'NoMask'}

        cap =cam
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        # writer = cv2.VideoWriter(output_video_name, fourcc, int(fps), (int(width), int(height)))
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if not cap.isOpened():
            raise ValueError("Video open failed.")
        status = True
        idx = 0
        while status:
            start_stamp = time.time()
            status, img_raw = cap.read()
            img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            read_frame_stamp = time.time()
            if (status):
                output_info = []
                height , width, _ = img_raw.shape
                image_resized = cv2.resize(img_raw, (360, 360))
                image_np = image_resized / 255.0  # 
                image_exp = np.expand_dims(image_np, axis=0)
                image_transposed = image_exp.transpose((0, 3, 1, 2))
                y_bboxes_output, y_cls_output= pytorch_inference(model, image_transposed)
                # remove the batch dimension, for batch is always 1 for inference.
                y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
                y_cls = y_cls_output[0]
                # To speed up, do single class NMS, not multiple classes NMS.
                bbox_max_scores = np.max(y_cls, axis=1)
                bbox_max_score_classes = np.argmax(y_cls, axis=1)

                # keep_idx is the alive bounding box after nms.
                keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                            bbox_max_scores,
                                                            conf_thresh=0.5,
                                                            iou_thresh=0.5,
                                                            )

                for idx in keep_idxs:
                    conf = float(bbox_max_scores[idx])
                    class_id = bbox_max_score_classes[idx]
                    bbox = y_bboxes[idx]
                    # clip the coordinate, avoid the value exceed the image boundary.
                    xmin = max(0, int(bbox[0] * width))
                    ymin = max(0, int(bbox[1] * height))
                    xmax = min(int(bbox[2] * width), width)
                    ymax = min(int(bbox[3] * height), height)

                    draw_result=True
                    if draw_result:
                        if class_id == 0:
                            color = (0, 255, 0)
                        else:
                            color = (255, 0, 0)
                        cv2.rectangle(img_raw, (xmin, ymin), (xmax, ymax), color, 2)
                        
                        cv2.putText(img_raw, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
                    output_info.append([class_id, conf, xmin, ymin, xmax, ymax])
                cv2.imshow('image', img_raw[:, :, ::-1])
                cv2.waitKey(1)
                inference_stamp = time.time()
                        # writer.write(img_raw)
                write_frame_stamp = time.time()
                idx += 1
                print("%d of %d" % (idx, total_frames))
                print("read_frame:%f, infer time:%f, write time:%f" % (read_frame_stamp - start_stamp,
                                                                            inference_stamp - read_frame_stamp,
                                                                           write_frame_stamp - inference_stamp))
   
        
            im=cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
            ret,jpg=cv2.imencode('.jpg',im)
                
            return jpg.tobytes()
