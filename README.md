# MSc-Dissertation

This Repository contains several simple steps to process and assess finger functions (1.Final App.ipynb), simple comparision of CNNs (2.CNN comparision.ipynb) and 13 CNN models and traditional ML models (.py files).

## Related libraries used in this project are: 
Python == 3.8.8
CV2 == 4.6.0
numpy == 1.20.1
matplotlib == 3.3.4
sklearn == 0.24.1
PIL == 8.2.0
keras == 2.9.0
skimage == 0.18.1
pandas == 1.2.4

## Steps of grading fingers' function (1.Fianl App.ipynb)
1. injured and normal hand images are in "images" folder
2. load trained model "CNN_DenseNet169.h5". this large model can't be uploaded (you can refer to DenseNet169.py)
3. input data is segmented finger images (resize to 512x128x3 per image) and the output is six classes (6 Grades)
4. these finger images (index, middle, ring and little) are cropped from a frontal-hand image by using Hand Mediapipe annotation sysyem to locate the area of each finger.
4-1. Load image
![image](https://user-images.githubusercontent.com/26786836/197940287-2366b6fe-2e9b-4200-ae16-37d26cc66c86.png)
4-2. Annotate finger areas via Hand Mediapipe
![image](https://user-images.githubusercontent.com/26786836/197940451-0e84ae4d-94bc-4308-878c-2f67ceab1f19.png)
4-3. Cropped finger images for assessment (prediction)
![image](https://user-images.githubusercontent.com/26786836/197940585-a3992391-1cea-45d4-af0a-f955dec6327f.png)
5. input these images to the CNN model and get the prediction, and then show the results on the hand image
![image](https://user-images.githubusercontent.com/26786836/197940817-801bb189-c619-4c62-b8f1-36a32bcd6032.png)
6. in this case, the little finger get the wrong result due to the black lines crossing on the finger.


