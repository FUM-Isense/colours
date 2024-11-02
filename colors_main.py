import pyrealsense2 as rs
from pynput import keyboard
import time
import numpy as np
import cv2
from TextToSpeechPlayer import TextToSpeechPlayer

tts = TextToSpeechPlayer()
start = False

def on_press(key):
    global start
    try:
        if key.char == 'b':  # Detect when 'b' is pressed
            start = True
    except AttributeError:
        pass

class FrameCapturing:

    def __init__(self):
        
        # Create a pipeline
        self.pipeline = rs.pipeline()

        # Create a configuration object
        config = rs.config()

        # Enable the stream to record from both depth and color
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start streaming with the specified configuration
        self.pipeline.start(config)
        # device = pipeline.get_active_profile().get_device()
        # color_sensor = device.query_sensors()[1]
        # color_sensor.set_option(rs.option.enable_auto_exposure, False)

    def capture_frame(self):
        try:
            while True:
                # Wait for a coherent pair of frames (depth and color)
                frames = self.pipeline.wait_for_frames()

                # Get the color frame
                color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                # Convert the color frame to a NumPy array for OpenCV
                np_frame  = np.asanyarray(color_frame.get_data())
                color_image = cv2.rotate(np_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                return color_image

        except KeyboardInterrupt:
            print("\nRecording stopped by the user.")

    def stop(self):
        # Release resources
        self.pipeline.stop()


class ColorDetection:
    def __init__(self):
        self.hue_list = []
        self.decision = []
        self.colors = {
            'names': ['BLUE', 'GREEN', 'RED', 'CYAN', 'MAGENTA'],
            'hsv': [120, 60, 0, 90, 150],
            'lab': [-58, 134, 40, 163, -32]
        }
        self.next_flag = False

    def feed_image(self, img):
        h,w,_ = img.shape
        h1_ = h//2-h//10
        h2_ = h//2+h//10
        w1_ = w//2-w//10
        w2_ = w//2+w//10   

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        cbcr = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        patch = img[h1_:h2_, w1_:w2_]
        # cv2.imshow('patch', patch)

        if np.var(lab[...,0][h1_:h2_, w1_:w2_])<50:
            self.hue_list.append([np.mean(lab[...,0][h1_:h2_, w1_:w2_]), np.mean(lab[...,1][h1_:h2_, w1_:w2_]),np.mean(lab[...,2][h1_:h2_, w1_:w2_]), np.mean(hsv[...,0][h1_:h2_, w1_:w2_]), np.mean(hsv[...,-1][h1_:h2_, w1_:w2_]),
                            np.mean(patch[...,0]), np.mean(patch[...,1]), np.mean(patch[...,2]), np.mean(cbcr[...,1][h1_:h2_, w1_:w2_]), np.mean(cbcr[...,2][h1_:h2_, w1_:w2_])])
        
        elif np.var(hsv[...,-1][h1_:h2_, w1_:w2_])>1000:
            if len(self.hue_list)>5:
                tmp_decision = []
                
                diff = [np.abs(np.mean(self.hue_list, axis=0)[-5]-np.mean(self.hue_list, axis=0)[-4]), np.abs(np.mean(self.hue_list, axis=0)[-5]-np.mean(self.hue_list, axis=0)[-3]), np.abs(np.mean(self.hue_list, axis=0)[-4]-np.mean(self.hue_list, axis=0)[-3])]
                ii = np.argmin(diff)
                if ii == 0:
                    cluster_centers_ = [(np.mean(self.hue_list, axis=0)[-5]+np.mean(self.hue_list, axis=0)[-4])/2, np.mean(self.hue_list, axis=0)[-3]]
                    cluster_labels_ = [0, 0, 1]
                elif ii==1:
                    cluster_centers_ = [(np.mean(self.hue_list, axis=0)[-5]+np.mean(self.hue_list, axis=0)[-3])/2, np.mean(self.hue_list, axis=0)[-4]]
                    cluster_labels_ = [0, 1,  0 ]
                elif ii==2:
                    cluster_centers_ = [(np.mean(self.hue_list, axis=0)[-4]+np.mean(self.hue_list, axis=0)[-3])/2, np.mean(self.hue_list, axis=0)[-5]]
                    cluster_labels_ = [1, 0, 0]

                tmp_decision.append(np.argmin(np.abs(self.colors['hsv'] - np.mean(self.hue_list, axis=0)[3])))
                tmp_decision.append(np.argmin(np.abs(self.colors['lab'] - np.rad2deg(np.arctan2((np.mean(self.hue_list, axis=0)[2]-127),(np.mean(self.hue_list, axis=0)[1]-127))))))
                if np.sum(cluster_labels_==np.argmax(cluster_centers_))==1:
                    if (cluster_labels_==np.argmax(cluster_centers_))[0]:
                        tmp_decision.append(0)
                    elif (cluster_labels_==np.argmax(cluster_centers_))[1]:
                        tmp_decision.append(1)
                    elif (cluster_labels_==np.argmax(cluster_centers_))[2]:
                        tmp_decision.append(2)
                elif np.sum((cluster_labels_==np.argmax(cluster_centers_)))==2:
                    if (cluster_labels_==np.argmax(cluster_centers_))[0] and (cluster_labels_==np.argmax(cluster_centers_))[1]:
                        tmp_decision.append(3)
                    elif (cluster_labels_==np.argmax(cluster_centers_))[0] and (cluster_labels_==np.argmax(cluster_centers_))[2]:
                        tmp_decision.append(4)
                print(tmp_decision)
                diff_score = 0
                if len(sorted(set(tmp_decision), key=tmp_decision.count, reverse=True)) > 1:
                    if sorted(set(tmp_decision), key=tmp_decision.count, reverse=True)[1] == np.argmin(np.abs(self.colors['hsv'] - np.mean(self.hue_list, axis=0)[3])):
                        diff_score = np.min(np.abs(self.colors['hsv'] - np.mean(self.hue_list, axis=0)[3]))
                    elif sorted(set(tmp_decision), key=tmp_decision.count, reverse=True)[1] == np.argmin(np.abs(self.colors['lab'] - np.rad2deg(np.arctan2((np.mean(self.hue_list, axis=0)[2]-127),(np.mean(self.hue_list, axis=0)[1]-127))))):
                        diff_score = np.min(np.abs(self.colors['lab'] - np.rad2deg(np.arctan2((np.mean(self.hue_list, axis=0)[2]-127),(np.mean(self.hue_list, axis=0)[1]-127)))))
                    else:
                        diff_score = np.abs(self.colors['hsv'][sorted(set(tmp_decision), key=tmp_decision.count, reverse=True)[1]] - self.colors['hsv'][sorted(set(tmp_decision), key=tmp_decision.count, reverse=True)[0]])
                self.decision.append([max(set(tmp_decision), key=tmp_decision.count), np.mean(self.hue_list, axis=0)[-4]+np.mean(self.hue_list, axis=0)[-3]+np.mean(self.hue_list, axis=0)[-5], len(self.decision)+1, sorted(set(tmp_decision), key=tmp_decision.count, reverse=True), diff_score])

            self.hue_list = []
            self.next_flag = True

        return self.__next()

    def __next(self):
        if len(self.decision)==6:
            self.post_process(self.decision)
            return False

        if len(self.hue_list)>5 and self.next_flag:
            tts.say("next")
            self.next_flag = False

        return True

    def post_process(self, decision):
        positions = [0]*6
        cur = 1
        number_of_colors = [0]*5
        for el in decision:
            number_of_colors[el[0]]+=1


        check = False
        try:
            if 3 in number_of_colors:
                check = True
                if 2 in number_of_colors:
                    idx1 = number_of_colors.index(1)
                    idx2 = number_of_colors.index(2)
                    for el in decision:
                        if el[0]==idx1:
                            el[0]=idx2
            elif 2 in number_of_colors:
                if 1 in number_of_colors:
                    check = True
                    idx = np.where(np.array(number_of_colors)==2)[0]
                    err = np.where(np.array(number_of_colors)==1)[0]
                    diff = [[np.abs(self.colors['hsv'][err[0]]-self.colors['hsv'][idx[0]]), np.abs(self.colors['hsv'][err[0]]-self.colors['hsv'][idx[1]])], [np.abs(self.colors['hsv'][err[1]]-self.colors['hsv'][idx[0]]), np.abs(self.colors['hsv'][err[1]]-self.colors['hsv'][idx[1]])]]
                    ii = np.argmin(diff)
                    for el in decision:
                        if el[0] == err[ii//2]:
                            el[0] = idx[ii%2]
                        if el[0] == err[(1-(ii//2))]:
                            el[0] = idx[(1-(ii%2))]
        except:
            check = True

        if not check:
            if number_of_colors[2]==0 and number_of_colors[4]==0:   # no red or magenta detected and should just be blue and green
                should_be = [3,3,0,0,0]
                for el in decision:
                    if len(el[3]) == 1:
                        if el[3][0] == 0:   #blue
                            should_be[0]-=1
                        elif el[3][0] == 1:
                            should_be[0]-=1
                if should_be[0] > 0:
                    first_options = []
                    for idx, el in enumerate(decision):
                        if len(el[3]) == 2 and el[3][0]==0:
                            first_options.append((idx, el[4]))
                    sorted_first_options = sorted(first_options, key=lambda x:x[1], reverse=True)
                    for el in sorted_first_options[:should_be[0]]:
                        decision[el[0]][0] = 0
                    should_be[0]-=len(sorted_first_options)
                if should_be[0] > 0:
                    second_options = []
                    for idx, el in enumerate(decision):
                        if len(el[3]) == 2 and el[3][1] == 0:
                            second_options.append((idx, el[4]))
                    sorted_second_options = sorted(second_options, key=lambda x:x[1])
                    for el in sorted_second_options[:should_be[0]]:
                        decision[el[0]][0] = 0
                if should_be[1] > 0:
                    first_options = []
                    for idx, el in enumerate(decision):
                        if len(el[3]) == 2 and el[3][0]==1:
                            first_options.append((idx, el[4]))
                    sorted_first_options = sorted(first_options, key=lambda x:x[1], reverse=True)
                    for el in sorted_first_options[:should_be[1]]:
                        decision[el[0]][0] = 1
                    should_be[1]-=len(sorted_first_options)
                if should_be[1] > 0:
                    second_options = []
                    for idx, el in enumerate(decision):
                        if len(el[3]) == 2 and el[3][1] == 1:
                            second_options.append((idx, el[4]))
                    sorted_second_options = sorted(second_options, key=lambda x:x[1])
                    for el in sorted_second_options[:should_be[1]]:
                        decision[el[0]][0] = 1
            else:
                determined_options = [0]*5
                for idx, el in enumerate(decision):
                    if len(el[3]) == 1:
                        determined_options[el[3][0]]+=1
                while np.sum(determined_options)>0:
                    first_choice = np.argmax(determined_options)
                    if determined_options[first_choice] < 3:
                        first_options = []
                        for idx, el in enumerate(decision):
                            if len(el[3]) == 2 and el[3][0]==first_choice:
                                first_options.append((idx, el[4]))
                        sorted_first_options = sorted(first_options, key=lambda x:x[1], reverse=True)
                        for el in sorted_first_options[:(3-determined_options[first_choice])]:
                            decision[el[0]][0] = first_choice
                        determined_options[first_choice]+=len(sorted_first_options[:(3-determined_options[first_choice])])
                    if determined_options[first_choice] < 3:
                        second_options = []
                        for idx, el in enumerate(decision):
                            if len(el[3]) == 2 and el[3][1] == first_choice:
                                second_options.append((idx, el[4]))
                        sorted_second_options = sorted(second_options, key=lambda x:x[1])
                        for el in sorted_second_options[:(3-determined_options[first_choice])]:
                            decision[el[0]][0] = first_choice
                    determined_options[first_choice]=0

            check = False
            try:
                if 3 in number_of_colors:
                    check = True
                    if 2 in number_of_colors:
                        idx1 = number_of_colors.index(1)
                        idx2 = number_of_colors.index(2)
                        for el in decision:
                            if el[0]==idx1:
                                el[0]=idx2
                elif 2 in number_of_colors:
                    if 1 in number_of_colors:
                        check = True
                        idx = np.where(np.array(number_of_colors)==2)[0]
                        err = np.where(np.array(number_of_colors)==1)[0]
                        diff = [[np.abs(self.colors['hsv'][err[0]]-self.colors['hsv'][idx[0]]), np.abs(self.colors['hsv'][err[0]]-self.colors['hsv'][idx[1]])], [np.abs(self.colors['hsv'][err[1]]-self.colors['hsv'][idx[0]]), np.abs(self.colors['hsv'][err[1]]-self.colors['hsv'][idx[1]])]]
                        ii = np.argmin(diff)
                        for el in decision:
                            if el[0] == err[ii//2]:
                                el[0] = idx[ii%2]
                            if el[0] == err[(1-(ii//2))]:
                                el[0] = idx[(1-(ii%2))]
            except:
                check = True


            if not check:
                tts.say('do it again')
        else:
            for i in range(6):
                for el in [x  for x in sorted(decision, key=lambda x:x[1]) if x[0]==i]:
                    positions[el[2]-1] = cur
                    cur+=1

            number = 0
            while len(positions)>0:
                number*=10
                number+=(np.argmin(positions)+1)
                del positions[np.argmin(positions)]
            print(number)
            tts.say(str(number//1000))
            tts.say(str(number%1000))
            time.sleep(1)


if __name__ == '__main__':
    camera = FrameCapturing()
    detection = ColorDetection()


    # Start keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()
    
    while True:
        frame = camera.capture_frame()

        # Display the color frame in a window
        # cv2.imshow('RealSense Color Frame', frame)

        if start:
            if not detection.feed_image(frame):
                break

        cv2.waitKey(1)
    
    tts.stop()
    camera.stop()
    cv2.destroyAllWindows()
    listener.stop()
