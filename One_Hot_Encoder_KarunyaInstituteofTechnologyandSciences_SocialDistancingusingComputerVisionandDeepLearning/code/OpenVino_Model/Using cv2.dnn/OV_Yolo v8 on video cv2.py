INPUT_WIDTH = 640
INPUT_HEIGHT = 640

# initialize minimum probability to filter weak detections along with
CONFIDENCE_THRESHOLD = 0.1

# the threshold when applying non-maximum suppression
NMS_THRESHOLD = 0.4

# define the minimum safe distance factor that two people can be
# from each other
MIN_DISTANCE_FACTOR = 1.3


import cv2
import time
import numpy as np


def plot_boxes(img:np.ndarray, results:np.ndarray = None, color:tuple[int, int, int] = [0, 255, 0], boxes =None, line_thickness:int = None, defaulters:list[int] = None):
    if results is None and boxes is None:
        print("Error: results and centroids are both None, at least one of them should be not None")
        return
    bboxes= boxes or results[0]
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    for idx, box in enumerate(bboxes):
        if defaulters and idx in defaulters:
            color = [255, 255, 0]
        else:
            color = [0, 255, 0]
        c1, c2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)


def calc_centroids(results:np.ndarray):
    bboxes= results[0]
    centroids = []
    h_approx = []
    for box in bboxes:
        #box = xyxy[0:4]
        c = int((box[0] + box[2])/2), int((box[1] + box[3])/2)
        h = abs(box[2]-box[0])
        centroids.append(c)
        h_approx.append(h)

    # Ideal Min Distance for social distancing is 6ft or 2m
    #So we return height of the humans to get a estimate of ~6ft in pixels at that point
    return centroids, h_approx


def plot_centroids(img:np.ndarray, results:np.ndarray = None, color:tuple[int, int, int] = [0, 255, 0], centroids:list[tuple[int, int]] = None, line_thickness:int = None, defaulters:list[int] = None):
    if results is None and centroids is None:
        print("Error: results and centroids are both None, at least one of them should be not None")
        return
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    centroids = centroids or calc_centroids(results)[0]
    for idx, centroid in enumerate(centroids):
        if defaulters and idx in defaulters:
            color = [255, 255, 0]
        else:
            color = [0, 255, 0]
        cv2.circle(img =img, center =centroid,radius=3, color=color, thickness=tl)


def show_violations(img:np.ndarray, instances:list[int], results:np.ndarray = None, centroids:list[tuple[int, int]]= None, line_thickness:int = None, color = [0, 255, 0]):
    if results is None and centroids is None:
        print("Error: results and centroids are both None, at least one of them should be not None")
        return
    elif centroids is None:
        centroids = calc_centroids(results)[0]

    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    for violation in instances:
        color = [255, 0, 0]
        i, j = violation
        cv2.line(img, centroids[i], centroids[j], color, tl)
    
    # draw the total number of social distancing violations on the output frame
    text = "Social Distancing Violations: {}".format(len(instances))
    cv2.putText(img, text, (10, img.shape[0] - 25),
		cv2.FONT_HERSHEY_SIMPLEX, 1.00, color, tl)
    


from scipy.spatial import distance as dist
def check_violations (results:dict, source_image:np.ndarray):
    violators = set()
    violations = set()
    if len(results[0]) >= 2:
		# extract all centroids from the results and compute the
		# Euclidean distances between all pairs of the centroids
        centroids, min_dist = calc_centroids(results)
        D = dist.cdist(centroids, centroids, metric="euclidean")

		# loop over the upper triangular of the distance matrix
        for i in range(0, D.shape[0]):
            for j in range(i + 1, D.shape[1]):
				# check to see if the distance between any two
				# centroid pairs is less than the configured number
				# of pixels
                if D[i, j] < ((min_dist[i]+min_dist[j])/2)*MIN_DISTANCE_FACTOR:
					# update our violation set with the indexes of
					# the centroid pairs
                    violators.add(i)
                    violators.add(j)
                    violations.add((i, j))
	
	# return the set of violations
    return violators, violations

		

def draw_results(results:dict, source_image:np.ndarray):
    """
    Helper function for drawing bounding boxes on image
    Parameters:
        image (np.ndarray): for drawing predictions of the format [x1, y1, x2, y2]
        source_image (np.ndarray): input image for drawing bounding boxes and centroids
    Returns:    
    """
    # boxes = results[0]
    # centroids = calc_centroids(results)[0]
    # print(centroids)

    violators, violations = check_violations(results, source_image)
    
    # Plot the bounding boxes of the people detected
    plot_boxes(source_image, results, defaulters=violators)#, boxes=boxes)

    # Plot the centroids of the people detected
    plot_centroids(source_image, results, defaulters=violators)#, centroids=centroids)

    # Plot lines between centroids of people violating social distancing
    show_violations(source_image, instances= violations, results=results)#, centroids=centroids)



# Interfacing model outpurs with the rest of the code
def extract_boxes(preds:tuple, img:np.ndarray):
    confs, boxes = list(), list()

    image_height, image_width, _ = img.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT

    rows = preds[0].shape[0]

    for i in range(rows):
        row = preds[0][i]
        conf = row[4]
        
        classes_score = row[4:]
        _,_,_, max_idx = cv2.minMaxLoc(classes_score)

        if (max_idx[0] == 0):

            if (classes_score[0] > .25):
                confs.append(conf)
                
                #extract boxes
                x, y, w, h = row[0:4] 
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, left+width, top+height])
                boxes.append(box)
        else:
            continue
            
    r_confs, r_boxes = list(), list()

    indexes = cv2.dnn.NMSBoxes(boxes, confs, 0.25, 0.45) 
    for i in indexes:
        r_confs.append(confs[i])
        r_boxes.append(boxes[i])
        #print('person', confs[i], boxes[i])
    return r_boxes, r_confs


def show_fps(frame, prev_frame_time=0):
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = "FPS:"+str(round(fps))
    # putting the FPS count on the frame
    cv2.putText(frame, fps, (frame.shape[1] - 100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (100, 255, 0), 3, cv2.LINE_AA)
    return prev_frame_time


v1 = "people-detection.mp4"
v2 = "pedestrians.mp4"
#Change path to input and output name
vid_name = v1
path = "data/"+vid_name

# Openvino runtime using cv2.dnn
# Load a model
net = cv2.dnn.readNet('models/yolov8n.onnx')  # load a pretrained model

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(path)

prev_frame_time = 0

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
size = (int(cap.get(3)), int(cap.get(4)))
video_writer = cv2.VideoWriter(
    'demo_videos/OpenVino_Model/Using cv2.dnn/out_'+vid_name, 
    fourcc, 10.0, size)

while (cap.isOpened()):
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    preds = preds.transpose((0, 2, 1))

    results = extract_boxes(preds, frame)
    draw_results(results, frame) # draw the results
    prev_frame_time = show_fps(frame, prev_frame_time) # show fps
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("Openvino using cv2.dnn", frame)
    video_writer.write(frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# When everything done, release
# the video capture and video
# write objects
cap.release()
video_writer.release()
	
# Closes all the frames
cv2.destroyAllWindows()

print("The video was successfully saved")