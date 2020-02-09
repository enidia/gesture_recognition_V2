import tensorflow as tf
from tensorflow import keras
import numpy as np
from utils import tracking_module, utils
import cv2
import math
import importlib
import os

from config import FLAGS

cpm_model = importlib.import_module('models.nets.' + FLAGS.network_def)
model_path = './models/weights/106save03.h5'

class runTracker(object):
    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu_id)
        self.init_value()

    def init_value(self):
        self.joint_detections = np.zeros(shape=(21, 2))
        # Initial tracker
        self.tracker = tracking_module.SelfTracker([FLAGS.webcam_height, FLAGS.webcam_width], FLAGS.input_size)

        # Build network graph
        self.model = cpm_model.CPM_Model(input_size=FLAGS.input_size,
                                heatmap_size=FLAGS.heatmap_size,
                                stages=FLAGS.cpm_stages,
                                joints=FLAGS.num_of_joints,
                                img_type=FLAGS.color_channel,
                                is_training=False)

        self.saver = tf.compat.v1.train.Saver()

        # Get output node
        self.output_node = tf.compat.v1.get_default_graph().get_tensor_by_name(name=FLAGS.output_node_names)

        device_count = {'GPU': 1} if FLAGS.use_gpu else {'GPU': 0}
        sess_config = tf.compat.v1.ConfigProto(device_count=device_count)
        sess_config.gpu_options.per_process_gpu_memory_fraction = 0.7
        sess_config.gpu_options.allow_growth = True
        sess_config.allow_soft_placement = True
        self.sess = tf.compat.v1.Session(config=sess_config)

        model_path_suffix = os.path.join(FLAGS.network_def,
                                         'input_{}_output_{}'.format(FLAGS.input_size, FLAGS.heatmap_size),
                                         'joints_{}'.format(FLAGS.num_of_joints),
                                         'stages_{}'.format(FLAGS.cpm_stages),
                                         'init_{}_rate_{}_step_{}'.format(FLAGS.init_lr, FLAGS.lr_decay_rate,
                                                                          FLAGS.lr_decay_step)
                                         )
        model_save_dir = os.path.join('models',
                                      'weights',
                                      model_path_suffix)
        print('Load model from [{}]'.format(os.path.join(model_save_dir, FLAGS.model_path)))
        self.saver.restore(self.sess, 'models/weights/cpm_hand')

        # new model

        self.new_model = keras.models.load_model(model_path)
        self.new_model.summary()

        self.__CreateKalmanfilters()

        self.cam = cv2.VideoCapture(FLAGS.cam_id)

    def isOpenVideoCap(self):
        return self.cam.isOpened()

    # Create kalman filters
    def __CreateKalmanfilters(self):
        if FLAGS.use_kalman:
            self.kalman_filter_array = [cv2.KalmanFilter(4, 2) for _ in range(FLAGS.num_of_joints)]
            for _, joint_kalman_filter in enumerate(self.kalman_filter_array):
                joint_kalman_filter.transitionMatrix = np.array(
                    [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
                    np.float32)
                joint_kalman_filter.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
                joint_kalman_filter.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                                                               np.float32) * FLAGS.kalman_noise
        else:
            self.kalman_filter_array = None

    # Create Joint Img
    def CreateImg(self):
        if (self.cam.isOpened() !=  True):
            print('failed')
            return None, None

        _, full_img = self.cam.read()
        test_img = self.tracker.tracking_by_joints(full_img, joint_detections=self.joint_detections)
        crop_full_scale = self.tracker.input_crop_ratio
        test_img_copy = test_img.copy()

        # White balance
        test_img_wb = utils.img_white_balance(test_img, 5)
        test_img_input = self.__normalize_and_centralize_img(test_img_wb)

        # Inference
        stage_heatmap_np = self.sess.run([self.output_node],
                                    feed_dict={self.model.input_images: test_img_input})

        local_img, hf_img = self.__visualize_result(full_img, stage_heatmap_np, self.kalman_filter_array, self.tracker,
                                            crop_full_scale, test_img_copy)

        # cv2.imshow('local_img', local_img.astype(np.uint8))
        # cv2.imshow('global_img', full_img.astype(np.uint8))
        return full_img, local_img, hf_img

    #Predict the Img use newModel
    def PredictImg(self, local_img):
        dict = ["零","一","二","三","四","五","六","七","八","九","OK"]#self.__get_list('./storePic') #["up","ye", "stop", "666","eight","ok"]
        # print('get a GP')
        ret,temp = cv2.threshold(local_img,90,255,cv2.THRESH_BINARY) #erzhihua
        img = cv2.resize(temp, (100, 100))
        
        data = []
        mat = np.asarray(img)
        data.append(mat)
        data = np.asarray(data)

        output = self.new_model.predict(data)
        if np.max(output[0]) * 100 > 90:
            # print("the G P:" + dict[np.argmax(output[0])] + " num:{:2.0f}%".format(np.max(output[0]) * 100))
            return dict[np.argmax(output[0])], np.max(output[0]) * 100
        else:
            # print("G P fail")
            return "", 0.0


    def __get_list(self, path):
        dict = []
        cate = [path + '/' + x for x in os.listdir(path) if os.path.isdir(path + '/' + x)]
        for idx, folder in enumerate(cate):
            dirname = folder.split(path + '/')
            dict.append(dirname[1])
        return dict

    def __normalize_and_centralize_img(self, img):
        if FLAGS.color_channel == 'GRAY':
            img = np.dot(img[..., :3], [0.299, 0.587, 0.114]).reshape((FLAGS.input_size, FLAGS.input_size, 1))

        if FLAGS.normalize_img:
            test_img_input = img / 256.0 - 0.5
            test_img_input = np.expand_dims(test_img_input, axis=0)
        else:
            test_img_input = img - 128.0
            test_img_input = np.expand_dims(test_img_input, axis=0)
        return test_img_input

    def __visualize_result(self, test_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
        demo_stage_heatmaps = []
        for stage in range(len(stage_heatmap_np)):
            demo_stage_heatmap = stage_heatmap_np[stage][0, :, :, 0:FLAGS.num_of_joints].reshape(
                (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
            demo_stage_heatmap = cv2.resize(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size))
            demo_stage_heatmap = np.amax(demo_stage_heatmap, axis=2)
            demo_stage_heatmap = np.reshape(demo_stage_heatmap, (FLAGS.input_size, FLAGS.input_size, 1))
            demo_stage_heatmap = np.repeat(demo_stage_heatmap, 3, axis=2)
            demo_stage_heatmap *= 255
            demo_stage_heatmaps.append(demo_stage_heatmap)

        last_heatmap = stage_heatmap_np[len(stage_heatmap_np) - 1][0, :, :, 0:FLAGS.num_of_joints].reshape(
            (FLAGS.heatmap_size, FLAGS.heatmap_size, FLAGS.num_of_joints))
        last_heatmap = cv2.resize(last_heatmap, (FLAGS.input_size, FLAGS.input_size))

        self.__correct_and_draw_hand(test_img, last_heatmap, kalman_filter_array, tracker, crop_full_scale, crop_img)

        if len(demo_stage_heatmaps) > 3:
            upper_img = np.concatenate((demo_stage_heatmaps[0], demo_stage_heatmaps[1], demo_stage_heatmaps[2]), axis=1)
            lower_img = np.concatenate(
                (demo_stage_heatmaps[3], demo_stage_heatmaps[len(stage_heatmap_np) - 1], crop_img),
                axis=1)
            demo_img = np.concatenate((upper_img, lower_img), axis=0)
            return demo_img, crop_img
        else:
            return demo_stage_heatmaps[0], crop_img

    def __correct_and_draw_hand(self, full_img, stage_heatmap_np, kalman_filter_array, tracker, crop_full_scale, crop_img):
        joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))
        local_joint_coord_set = np.zeros((FLAGS.num_of_joints, 2))

        mean_response_val = 0.0

        # Plot joint colors
        if kalman_filter_array is not None:
            for joint_num in range(FLAGS.num_of_joints):
                tmp_heatmap = stage_heatmap_np[:, :, joint_num]
                joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                               (FLAGS.input_size, FLAGS.input_size))
                mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
                joint_coord = np.array(joint_coord).reshape((2, 1)).astype(np.float32)
                kalman_filter_array[joint_num].correct(joint_coord)
                kalman_pred = kalman_filter_array[joint_num].predict()
                correct_coord = np.array([kalman_pred[0], kalman_pred[1]]).reshape((2))
                local_joint_coord_set[joint_num, :] = correct_coord

                # Resize back
                correct_coord /= crop_full_scale

                # Substract padding border
                correct_coord[0] -= (tracker.pad_boundary[0] / crop_full_scale)
                correct_coord[1] -= (tracker.pad_boundary[2] / crop_full_scale)
                correct_coord[0] += tracker.bbox[0]
                correct_coord[1] += tracker.bbox[2]
                joint_coord_set[joint_num, :] = correct_coord

        else:
            for joint_num in range(FLAGS.num_of_joints):
                tmp_heatmap = stage_heatmap_np[:, :, joint_num]
                joint_coord = np.unravel_index(np.argmax(tmp_heatmap),
                                               (FLAGS.input_size, FLAGS.input_size))
                mean_response_val += tmp_heatmap[joint_coord[0], joint_coord[1]]
                joint_coord = np.array(joint_coord).astype(np.float32)

                local_joint_coord_set[joint_num, :] = joint_coord

                # Resize back
                joint_coord /= crop_full_scale

                # Substract padding border
                joint_coord[0] -= (tracker.pad_boundary[2] / crop_full_scale)
                joint_coord[1] -= (tracker.pad_boundary[0] / crop_full_scale)
                joint_coord[0] += tracker.bbox[0]
                joint_coord[1] += tracker.bbox[2]
                joint_coord_set[joint_num, :] = joint_coord

        self.__draw_hand(full_img, joint_coord_set, tracker.loss_track)
        #self.__draw_hand(crop_img, local_joint_coord_set, tracker.loss_track)
        self.joint_detections = joint_coord_set

        if mean_response_val >= 1:
            tracker.loss_track = False
        else:
            tracker.loss_track = True

        # cv2.putText(full_img, 'Response: {:<.3f}'.format(mean_response_val),
        #            org=(20, 20), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0))

    def __draw_hand(self, full_img, joint_coords, is_loss_track):
        if is_loss_track:
            joint_coords = FLAGS.default_hand

        # Plot joints
        for joint_num in range(FLAGS.num_of_joints):
            color_code_num = (joint_num // 4)
            if joint_num in [0, 4, 8, 12, 16]:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])),
                           radius=3,
                           color=joint_color, thickness=-1)
            else:
                joint_color = list(map(lambda x: x + 35 * (joint_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.circle(full_img, center=(int(joint_coords[joint_num][1]), int(joint_coords[joint_num][0])),
                           radius=3,
                           color=joint_color, thickness=-1)

        # Plot limbs
        for limb_num in range(len(FLAGS.limbs)):
            x1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][0])
            y1 = int(joint_coords[int(FLAGS.limbs[limb_num][0])][1])
            x2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][0])
            y2 = int(joint_coords[int(FLAGS.limbs[limb_num][1])][1])
            length = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
            if length < 150 and length > 5:
                deg = math.degrees(math.atan2(x1 - x2, y1 - y2))
                polygon = cv2.ellipse2Poly((int((y1 + y2) / 2), int((x1 + x2) / 2)),
                                           (int(length / 2), 3),
                                           int(deg),
                                           0, 360, 1)
                color_code_num = limb_num // 4
                limb_color = list(map(lambda x: x + 35 * (limb_num % 4), FLAGS.joint_color_code[color_code_num]))
                cv2.fillConvexPoly(full_img, polygon, color=limb_color)
    def __del__(self):
        self.__del__()