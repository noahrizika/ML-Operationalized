# serializeImageData

from sagemaker.serializers import IdentitySerializer
import sagemaker
import json
import boto3
import base64
import tempfile

s3 = boto3.client('s3')


def lambda_handler(event, context):
    """A function to serialize target data from S3"""

    # Get the s3 address from the Step Function event input
    key = event['s3_key']
    bucket = event['s3_bucket']

    # Download the data from s3 to /tmp/image.png
    image = '/tmp/image.png'
    s3.download_file(bucket, key, image)

    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }


# classifier

# Fill this in with the name of your deployed model
import os

ENDPOINT = os.environ.get('ENDPOINT_NAME')


def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event["image_data"])

    # Instantiate a Predictor
    predictor = sagemaker.predictor.Predictor(ENDPOINT)

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")

    # Make a prediction:
    inferences = predictor.predict(image)

    # We return the data back to the Step Function
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }


# filterInferences

THRESHOLD = .93


def lambda_handler(event, context):

    # Grab the inferences from the event

    # convert string to array of floats
    inferences = event["inferences"]
    inferences = inferences[1:-1]
    inferences = inferences.split()

    predictions = [0, 0]

    index = 0
    for x in inferences:
        predictions[index] = x.replace(",", "")
        predictions[index] = float(predictions[index])
        index = index + 1

    # Check if any values in our inferences are above THRESHOLD

    for prediction in predictions:
        if prediction >= THRESHOLD:
            meets_threshold = True
            break
        if prediction < THRESHOLD:
            meets_threshold = False

    # or...
    # meets_threshold = any(inference >= THRESHOLD for inference in inferences)

    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise ("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
