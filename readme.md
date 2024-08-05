# Project Overview: Mask Detection System

## Introduction
In today's world, ensuring safety through proper mask usage is paramount. Driven by the desire to create a purposeful and impactful solution, I have developed a system for stores and companies to detect individuals not wearing masks. This project leverages live camera feeds to identify maskless individuals in real-time, allowing for subsequent actions such as raising alarms.

## Development Journey

### 1. Fine-Tuning the Model
To achieve accurate mask detection, I fine-tuned the MobileNet CNN using a face mask detection dataset from Kaggle. This dataset comprised both synthetic and real images, totaling around 7,500 images. Post fine-tuning, the model achieved:
- Accuracy: 0.9269
- Precision: 0.9324
- Recall: 0.9184

### 2. Streamlit Integration
Streamlit was chosen for its simplicity and effectiveness in deploying ML models. The application consists of two pages:
- **Single Image Capture**: Users can capture and process a single image, with results displayed as a list.
- **Live Video Capture**: Using `streamlit_webrtc`, the application captures live video, processes frames every 0.4 seconds, and displays results as a list.

### 3. Image Processing Workflow
The image processing pipeline involves:
- **Face Detection**: Using the Haar Cascade algorithm to detect faces within an image.
- **Face Resizing**: Resizing detected faces for model prediction.
- **Model Prediction**: Passing resized faces to the model and mapping predictions to a list of boolean values.

In a production environment, any False value in the results can trigger a warning.

## Challenges and Solutions

### Video Frame Processing
A significant challenge was capturing video frames from the user camera and processing them in real-time. The `streamlit_webrtc` component, operating in a separate thread, posed threading issues when sending frames to the main Streamlit application thread.

To maintain clean and extendable code boundaries, it was crucial to process frames in the main thread. Through extensive learning and exploration of threading concepts, I devised a solution that balanced real-time processing with code maintainability.

### Improvements
To enhance the face detection system and reduce errors, several improvements can be made. One issue is that the model may predict "True" even if the user has a white cloth or similar object covering their face, mistaking it for a mask. Addressing this requires refining the dataset to include more diverse and challenging examples that can help the model better differentiate between actual masks and other coverings.

Another flaw is that the face recognition system sometimes detects faces in images where no faces are present. This issue can be mitigated by adjusting the strictness of the face detection algorithm, making it more discerning in identifying genuine faces.

## Conclusion
Despite the current model's limitations, such as occasional false positives with white cloths or similar objects and incorrect face detections, the results are reasonable and demonstrate significant potential for real-world application. This project provided a rich learning experience, deepening my understanding of fine-tuning convolutional neural networks, implementing real-time video processing, and managing threading issues in a web application context.

Throughout the development process, I honed my skills in maintaining clean and extendable code, ensuring that the system can be easily updated and scaled in the future. The challenges encountered, particularly with threading and real-time video frame processing, tested my problem-solving abilities and reinforced the importance of thorough testing and iterative improvement.

Looking ahead, I am excited to continue refining this system, incorporating feedback and new techniques to further enhance its accuracy and robustness. This project has not only contributed to my technical growth but also reaffirmed my commitment to creating impactful and purposeful solutions in the field of machine learning and computer vision. The journey has been both challenging and rewarding, and I am eager to see how this system evolves and benefits users in practical settings.