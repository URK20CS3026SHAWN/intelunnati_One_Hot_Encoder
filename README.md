# Social Distancing Project Using Computer Vision and Deep Learning

## Implementing Social Distancing using a pre-trained YOLO v8 Model:
1.	Load the YOLO v8 pre-trained model. There are many pre-trained YOLO v8 models available, so you can choose one that is appropriate for your needs.

2.	Use the model to detect people in real-time video. This can be done using a webcam or a security camera. The model will output the bounding boxes of the people it detects, as well as the confidence scores for each detection.

3.	Calculate the centroid for each bounding box. The centroid is the centre of the bounding box, and it can be used to calculate the distance between two people.

4.	Calculate the distance between the centroids of any two people. If the distance between two people is less than a threshold, then the model will flag them as a social distancing violation.

5.	Optional: Set the Distance Threshold as a function of the average human heights extracted from the bounding boxes.

6.	Now, the centroids and bounding boxes can be visualized along with lines that show the pairs of people violating the social distancing threshold.

### Regular Yolo v8 Output Screenshots
<img width="423" alt="image" src="https://github.com/URK20CS3026SHAWN/intelunnati_One_Hot_Encoder/assets/80960850/16210d51-3025-4e7f-b164-5d56894de04a">
<img width="423" alt="image" src="https://github.com/URK20CS3026SHAWN/intelunnati_One_Hot_Encoder/assets/80960850/c39def79-2470-47e9-9b66-b556ee4f07e3">

My laptop (M1 Series Chip)                                                      My colleague's Intel Laptop




## Optimizing the above model using the OpenVINO toolkit :
1.	Install the OpenVINO toolkit. You can download the OpenVINO toolkit from the Intel website: https://software.intel.com/en-us/openvino-toolkit.

2.	Download the YOLO v8 model. You can download the YOLO v8 model from the YOLO website: https://pjreddie.com/darknet/yolo/.

3.	Convert the YOLO v8 model to OpenVINO IR using the Model Optimizer tool.

4.	Implement the OpenVINO IR Optimized Model and make translation functions to augment the inputs and outputs, so that it can make use of the functions defined to implement social distancing using the Regular YOLO v8 model.

5.	Compare the outputs of the Models and Deploy a model that’s apt for the deployment environment and customer requirements.

6.	Benchmark the Performance of the YOLO v8 model and OpenVINO Optimized Model.

### Open Vino IR Optimized Model Output Screenshots
<img width="423" alt="image" src="https://github.com/URK20CS3026SHAWN/intelunnati_One_Hot_Encoder/assets/80960850/956ef6f5-12ca-49ce-ba44-c90e8b1bf8f0">
<img width="423" alt="image" src="https://github.com/URK20CS3026SHAWN/intelunnati_One_Hot_Encoder/assets/80960850/793159dd-2525-44e7-b2e5-17205a0000a7">










