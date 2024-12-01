from picamera2 import Picamera2
import libcamera
import cv2
from libcamera import controls

def take_image():
    pica=Picamera2()
    config = pica.create_still_configuration(
                                           main={"size":(1600,1920), 'format': 'RGB888'},
                                           raw = None,
                                           #controls = {"NoiseReductionMode": controls.draft.NoiseReductionModeEnum.Off},
                                           #colour_space = libcamera.ColorSpace('Srgb'),
                                           transform = libcamera.Transform(hflip=True,vflip=True))
                                            
    #config['colour_space'] = libcamera.ColorSpace.sRGB
    
    print("----------- Config ---------", config)
    pica.configure(config)
    pica.start()
    pica.capture_file('test_samp.jpg')
    pica.stop()
    #pica.start_and_capture_file('test_640x640.jpg')
    #pica.start_and_record_video('cam_pos_test.mp4', duration=7)
    print("Captured and saved the image --------------> test.jpg")
    img = cv2.imread('test_samp.jpg')
    resized_img = cv2.resize(img, (320, 480))
    #img_clr = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('test_samp_clr.jpg', resized_img)
    
    
take_image()
