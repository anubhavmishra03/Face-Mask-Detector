# Face-Mask-Detector
Built a Tkinter GUI app integrating a CNN model to detect face masks. Users upload images/videos, and the model identifies masked faces. Supports safety protocols in various settings.

The face mask detector built using deep learning CNN (Convolutional Neural Network) and Tkinter is a graphical application designed to detect whether a person is wearing a face mask or not. The detector utilizes a pre-trained CNN model, which has been trained on a dataset of images containing individuals with and without face masks. Tkinter, a Python library for creating GUI applications, is used to create the user interface for the detector.

The application allows users to upload an image or select a video/live feed to analyze for face mask detection. Upon selection, the application processes the input using the CNN model to identify faces and determine whether they are wearing masks. Detected faces are highlighted, and a label indicates whether a mask is present or not. The Tkinter GUI provides a user-friendly interface with buttons for image/video selection and display areas for the input and output.

The development process involves:
1. Collecting and preprocessing a dataset of face images with and without masks.
2. Training a CNN model on the dataset to learn patterns and features associated with masked and unmasked faces.
3. Integrating the trained model into a Tkinter-based GUI application.
4. Implementing functionalities for image/video input, face detection, mask detection, and result visualization.
5. Fine-tuning and optimizing the application for improved performance and user experience.

The face mask detector serves as a tool for enforcing safety protocols in various settings, such as public spaces, workplaces, and healthcare facilities, by quickly identifying individuals who may not be wearing masks. It demonstrates the synergy between deep learning techniques for image analysis and GUI development for creating practical and accessible applications.

**Dataset:**

With Mask: 2165

Without Mask: 1930

**Technology Used:**

1. Deep Learning
2. CNN
3. Tkinter
4. Jupyter
5. Opencv
6. Python

**Accuracy:**

MobileNetV2: 99.19

CNN: 98.18

VGG16: 94.2
