import streamlit as st
import tensorflow as tf
import numpy as np
from streamlit_option_menu import option_menu
from englisttohindi.englisttohindi import EngtoHindi
import time

st.set_page_config(page_title='Detect!t',page_icon="./letter-d.png",initial_sidebar_state="auto")


def model_prediction(test_image):
    model = tf.keras.models.load_model("testing3.h5",compile=False)
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

def preventive_measures(disease):
    measures = ""
    if disease == 'Greening-fruit':
        measures = ("1. Use reflective mulch to deter insect vectors.\n"
                    "2. Apply systemic insecticides to control psyllid populations.\n"
                    "3. Remove and destroy infected trees.\n"
                    "4. Regularly monitor trees for signs of infection.\n")
    elif disease == 'greening-leaf':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'Scab-fruit':
        measures = ("1. Remove and destroy infected leaves and fruit.\n"
                    "2. Prune trees for better airflow.\n"
                    "3. Apply fungicides during high disease pressure.\n"
                    "4. Use drip irrigation to minimize leaf wetness.\n"
                    "5. Consider planting scab-resistant varieties.")
    elif disease == 'Melanose-leaf':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'Black spot-leaf':
        measures = ("1. Prune infected branches regularly.\n"
                    "2. Keep foliage dry by watering at the base..\n"
                    "3. copper-based fungicides.\n"
                    "4. Remove fallen leaves and debris from around trees.\n")
    elif disease == 'black spot-fruit':
        measures = ("1. Remove infected plant material.\n"
                    "2. Apply copper sprays during outbreaks.\n"
                    "3. Prune infected branches below symptoms.\n"
                    "4. Minimize overhead irrigation.\n"
                    "5. Implement strict quarantine measures.")
    elif disease == 'citrus-canker-fruit':
        measures = ("1. Choose resistant varieties.\n"
                    "2. Remove infected plant material.\n"
                    "3. Apply copper sprays preventively.\n"
                    "4. Implement quarantine measures.\n"
                    "5. Monitor and detect symptoms early.")
    elif disease == 'canker-leaf':
        measures = ("1. Use reflective mulch to deter insect vectors.\n"
                    "2. Apply systemic insecticides to control psyllid populations.\n"
                    "3. Remove and destroy infected trees.\n"
                    "4. Regularly monitor trees for signs of infection.\n")
    else:
        measures = "No specific preventive measures found for the predicted disease."

    return measures

# 1. as sidebar menu
with st.sidebar:
    selected = option_menu("Main Menu", ["Home", 'About', 'Features','Disease Recognition'], 
        icons=['house', 'book','clipboard-data','search'], menu_icon="cast", default_index=0)
    selected

#Main pagerub
if (selected=="Home"): 
    st.header("Citrus Fruit Disease Detection Using Deep Learning")
    st.image("./home.jpg")
    st.markdown("""
    Welcome to the Citrus Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying citrus diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About
elif(selected=="About"):
    st.header("About")
    st.subheader("Introduction:")
    paragraph = """Citrus fruit detection plays a vital role in various industries, including agriculture and food processing. Accurately identifying and classifying citrus fruits such as oranges, lemons, and limes is crucial for quality control, sorting, and grading processes. Traditional methods of fruit inspection are time-consuming and labor-intensive, highlighting the need for automated detection systems.

In recent years, advancements in computer vision and machine learning have revolutionized fruit detection, enabling fast and accurate identification of citrus fruits based on their visual characteristics. These technologies have the potential to streamline fruit sorting processes, reduce waste, and improve overall efficiency in the citrus industry.

This project aims to develop a citrus fruit detection system using deep learning techniques. By leveraging the power of convolutional neural networks (CNNs) and image processing algorithms, we can create a robust solution capable of accurately detecting and classifying citrus fruits in real-time."""
    st.write(paragraph)
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this kaggle repo.
                This dataset consists of about 12K rgb images of healthy and diseased crop fruit and leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                Citrus Leaf Disease Image ( Orange Leaf)
                Types of images
                1. Black spot Leaf
                2. Canker Leaf
                3. Greening Leaf
                4. Healthy Leaf
                5. Melanose Leaf
                6. Scab Fruit
                7. Black Sot Fruit
                8. Greening Fruit
                9. Citrus Canker Fruit
                10. Healthy Fruit

                [Click here](https://www.kaggle.com/datasets/jonathansilva2020/dataset-for-classification-of-citrus-diseases) for Citrus fruit dataset
                
                [Click here](https://www.kaggle.com/datasets/myprojectdictionary/citrus-leaf-disease-image) for Citrus Leaf dataset

                #### Team Members
                """)
    team_members_html = """
    <div style="padding: 20px; background-color: #45A29E; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h4 style="margin-top: 0;">Abhishek Singh</h4>
        <p>🏫 College: Lokmanya Tilak College of Engineering</p>
        <p>🎓 Degree: Bachelor of Engineering</p>
        <p>🌳 Branch: Computer Engineering</p>
    </div>

    <div style="padding: 20px; background-color: #45A29E; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h4 style="margin-top: 0;">Himanshu Singh</h4>
        <p>🏫 College: Lokmanya Tilak College of Engineering</p>
        <p>🎓 Degree: Bachelor of Engineering</p>
        <p>🌳 Branch: Computer Engineering</p>
    </div>

    <div style="padding: 20px; background-color: #45A29E; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h4 style="margin-top: 0;">Manish Gupta</h4>
        <p>🏫 College: Lokmanya Tilak College of Engineering</p>
        <p>🎓 Degree: Bachelor of Engineering</p>
        <p>🌳 Branch: Computer Engineering</p>
    </div>

    <div style="padding: 20px; background-color: #45A29E; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 20px;">
        <h4 style="margin-top: 0;">Om Singh</h4>
        <p>🏫 College: Lokmanya Tilak College of Engineering</p>
        <p>🎓 Degree: Bachelor of Engineering</p>
        <p>🌳 Branch: Computer Engineering</p>
    </div>
    """

    st.markdown(team_members_html, unsafe_allow_html=True)
    


#Features
elif(selected=="Features"):
    st.header("Features")
    st.subheader("Features about the Project 📝🔍")
    feature_section_html = """
    <style>
    .feature {
        padding: 20px;
        background-color: #74C8C4;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        text-align: center;
    }

    .feature h3 {
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .feature img {
        width: 50px;
        height: 50px;
        margin-bottom: 10px;
        filter: invert(20%) sepia(100%) saturate(1500%) hue-rotate(180deg);
    }
    </style>

    <div class="feature">
        <img src="https://img.icons8.com/ios-filled/50/000000/upload--v1.png">
        <h3 style="color: #0078ff;">Easy Detection</h3>
        <p>Click the button below to upload an image of a citrus fruit for disease detection.</p>
    </div>

    <div class="feature">
        <img src="https://img.icons8.com/ios-filled/50/000000/idea-sharing.png">
        <h3 style="color: #ff6347;">Cause and Solution</h3>
        <p>After identifying a disease in a citrus fruit, the system can provide information on the cause of the disease and possible solutions or treatments.</p>
    </div>

    <div class="feature">
        <img src="https://img.icons8.com/ios-filled/50/000000/citrus.png">
        <h3 style="color: #3cba54;">Support for Different Citrus Fruit</h3>
        <p>This feature indicates that the system is capable of detecting diseases in various types of citrus fruits.</p>
    </div>
    """

    st.markdown(feature_section_html, unsafe_allow_html=True)
    

#Prediction
elif(selected=="Disease Recognition"):
    st.header("Disease Recognition")
    st.subheader("Test Your Fruit:")
    test_images = []

    option = st.selectbox('Choose an input Image option:',
                          ('--select option--','Upload', 'Camera'))
    
    if option == "Upload":
        test_images = st.file_uploader("Choose Image(s):", accept_multiple_files=True)
        if(st.button("Show Images")):
            st.image(test_images, width=4, use_column_width=True)

    elif option == "Camera":
        test_images = [st.camera_input("Capture an Image:")]
        if(st.button("Show Images")):
            st.image(test_images, width=4, use_column_width=True)


    if st.button("Predict"):
        for i, test_image in enumerate(test_images):
            st.write(f"Prediction for Image {i + 1}:")
            st.image(test_image, width=4, use_column_width=True)
            result_index = model_prediction(test_image)
            class_name = [
                "Black spot-leaf",
                "Greening-fruit",
                "Melanose-leaf",
                "Scab-fruit",
                "black-spot-fruit",
                "canker-leaf",
                "citrus-canker-fruit",
                "greening-leaf",
                "healthy-fruit",
                "healthy-leaf",
            ]
            predicted_class = class_name[result_index]

            if predicted_class == "healthy-fruit":
                st.success(f"Model is predicting it's a {predicted_class}")
            else:
                st.error(f"Model is predicting it's a {predicted_class}")

            co1, co2 = st.columns(2)
            with co1:
                st.write("Prevention Measures:")
                prevention_measures = preventive_measures(predicted_class)
                st.info(prevention_measures)

            with co2:
                res1 = EngtoHindi("Prevention Measures:")
                translated_measures = []
                for line in prevention_measures.strip().split("\n"):
                    res = EngtoHindi(line.strip())
                    translated_measures.append(res.convert)

                my_string = "\n".join(translated_measures)

                # Display the translated header and measures
                st.write(res1.convert)
                st.info(my_string)
