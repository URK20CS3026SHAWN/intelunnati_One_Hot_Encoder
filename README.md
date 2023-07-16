# intelunnati_One_Hot_Encoder

## Social Distancing Project Using Computer Vision and Deep Learning

### Implementing Social Distancing using a pretrained YOLO v8 Model:
1.	Load the YOLO v8 pretrained model. There are many pre-trained YOLO v8 models available, so you can choose one that is appropriate for your needs.

2.	Use the model to detect people in real-time video. This can be done using a webcam or a security camera. The model will output the bounding boxes of the people it detects, as well as the confidence scores for each detection.

3.	Calculate the centroid for each bounding box. The centroid is the center of the bounding box, and it can be used to calculate the distance between two people.

4.	Calculate the distance between the centroids of any two people. If the distance between two people is less than a threshold, then the model will flag them as a social distancing violation.

5.	Optional: Set the Distance Threshold as a function of the average human heights extracted from the bounding boxes.

6.	Now, the centroids and bounding boxes can be visualized along with lines that show the pairs of people violating the social distancing threshold.
7.	

### Optimizing the above model using OpenVINO toolkit :
1.	Install the OpenVINO toolkit. You can download the OpenVINO toolkit from the Intel website: https://software.intel.com/en-us/openvino-toolkit.

2.	Download the YOLO v8 model. You can download the YOLO v8 model from the YOLO website: https://pjreddie.com/darknet/yolo/.

3.	Convert the YOLO v8 model to OpenVINO IR using the Model Optimizer tool.

4.	Implement the OpenVINO IR Optimized Model and make translation functions to augment the inputs and outputs, so that it can make use of the functions defined to implement social distancing using the Regular YOLO v8 model.

5.	Compare the outputs of the Models and Deploy model that’s apt for the deployment environment and customer requirements.

6.	Benchmark the Performance of YOLO v8 model and OpenVINO Optimized Model.






