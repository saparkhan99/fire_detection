# fire_detection
Fire detection model is based on the yolov7.  The model was trained to differentiate two clases: smoke and fire with the dataset from kaggle.




# Model performances

The confusion matrix: 
![confusion_matrix](https://user-images.githubusercontent.com/47181212/231272843-0e594303-4980-4b18-8fc9-3dd4f629b200.png)

P_curve: 
![P_curve](https://user-images.githubusercontent.com/47181212/231273298-7cf301bf-3ea6-465d-9b63-c1a2b7f7c54d.png)

PR_curve:
![PR_curve](https://user-images.githubusercontent.com/47181212/231273174-e9ddda22-23ba-41e9-8d05-27f77cd27154.png)

R_curve:
![R_curve](https://user-images.githubusercontent.com/47181212/231273308-46a94895-4b36-4bee-8db9-596102c6d390.png)

The results:
![results](https://user-images.githubusercontent.com/47181212/231273184-5e1846a5-af63-4c50-b301-dba989d5371f.png)



Model vs. real predictions are in test folder.

# Weights
There are two types of weights are available. The first one in .pt format, which can be used for inference with basic yolo commands on terminal and onnx format used in the scripts provided.


# Testing

In order to run and test the model with real time detection, do the following:
  1. Download and use the yolov7 from the official github: https://github.com/WongKinYiu/yolov7
     Go through the instructions and intall the requirements.
  2. in your terminal run: "pip install onnxruntime"
  3. Download the onnx file.
  4. Download and run the scripts from this repository in the previously downloaded yolov7 folder.
  
