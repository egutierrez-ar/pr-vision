## PREPARATION ----------------------------------------------------------------------------------------------------------------------------------

# pip install azure-cognitiveservices-vision-customvision
# pip install opencv-python

## SETUP ----------------------------------------------------------------------------------------------------------------------------------------

from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
import cv2
import numpy as np
 
# Initialise PredictionClient that connects to the CustomVision API 
prediction_key = '24aa75256ea348568d5d44ca44839ce5'
ENDPOINT = "https://westeurope.api.cognitive.microsoft.com/"
predictor = CustomVisionPredictionClient(prediction_key, endpoint=ENDPOINT)

# The CustomVision API requires a project ID and model name so it knows which model to use
my_project_id = "7b94f2cd-28d8-4885-bc72-f72057f5869a" 
published_name = 'test'

# Initialise a Video Capture object for retrieving your camera images
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam.set(3, 1920)
cam.set(4, 1080)

## RUN APP -------------------------------------------------------------------------------------------------------------------------------------

# Run this loop continuously
while True:

    ### CAPTURE FRAME FROM CAMERA --------------------------------------------------------------------------------------------------------------

	# Read a single frame from your camera
    ret_val, img = cam.read()
    height, width, channels = img.shape

	# Save the frame as a png file
    cv2.imwrite('cam.png', img)

    ### PERFORM IMAGE RECOGNITION ON YOUR FRAME ------------------------------------------------------------------------------------------------

    # Open the image png file in binary format and send it to the CustomVision API. Results are immediately returned.
    with open("cam.png", mode="rb") as test_data:
        results = predictor.classify_image(my_project_id, published_name, test_data)

    ### INSERT CLASS PREDICTION IN VIDEO OUTPUT ------------------------------------------------------------------------------------------------

    # Check out the predictions
    for prediction in results.predictions:
        if prediction.probability > 0.5:
        	# Print the most probable class name in your terminal
            print("\t" + prediction.tag_name + ": {0:.2f}%".format(prediction.probability * 100))

            # Print the most probable class name in your video window
            text = prediction.tag_name + " ({0:.2f}%)".format(prediction.probability * 100)
            font_scale = 5
            font = cv2.FONT_HERSHEY_PLAIN

            # Set the rectangle background to white
            rectangle_bgr = (255, 255, 255)
            # Get the width and height of the text box
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]
            # Set the text start position
            text_offset_x = 10
            text_offset_y = img.shape[0] - 25
            # Make the coords of the box with a small padding of two pixels
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 20, text_offset_y - text_height - 20))
            cv2.rectangle(img, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
            cv2.putText(img, text, (text_offset_x, text_offset_y), font, fontScale=font_scale, color=(0, 0, 0), thickness=1)

    ### SHOW VIDEO OUTPUT -----------------------------------------------------------------------------------------------------------------------

    # Show video output including class prediction
    cv2.imshow('Beeldherkenning test - ProRail', img)
    
 	
 	# Stop application when Esc key is pressed
    if cv2.waitKey(1) == 27:
        break

# Stop all video processing after breaking out of the loop
cv2.destroyAllWindows()