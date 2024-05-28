from camera_utils import CameraUtils


camera_utils = CameraUtils(camera_id=0,
                           model_path='models/production/best.pt',
                           pred_dict={0: 'scratch'},
                           pred_display_color=(255, 150, 0),
                           display_size=[480, 640])

camera_utils.start_capture()
