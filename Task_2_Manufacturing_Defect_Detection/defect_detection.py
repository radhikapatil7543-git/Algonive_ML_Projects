import numpy as np
from keras.models import load_model
from keras.preprocessing import image

# Load trained model
model = load_model("defect_detection_model.keras")
print("\n========== DEFECT DETECTION RESULTS ==========\n")

def predict_image(image_path):
 
# Load and preprocess image
 img = image.load_img(image_path, target_size=(224,224))
 img_array = image.img_to_array(img)
 img_array = img_array / 255.0
 img_array = np.expand_dims(img_array, axis=0)

# Make prediction
 prediction = model.predict(img_array, verbose=0)
 probability = float(prediction[0][0])

# Display result
 if probability < 0.5:
    result = "Defective"
 else:
    result = "OK"

 print("Image:", image_path)
 print("Probability:", round(probability, 4))
 print("Prediction:", result)
 
 
# First image
predict_image("test_images/cast_def_0_108.jpeg")
print("-------------------------------------------")

# Second image
predict_image("test_images/cast_ok_0_1127.jpeg")
print("-------------------------------------------")
print("\n========== TESTING COMPLETED ==========\n")
