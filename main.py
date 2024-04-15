from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import cv2
import base64
from ultralytics import YOLO
import os
import smtplib
from email.message import EmailMessage

app = Flask(__name__)

# Define available models and their corresponding paths
models = {
    "YOLO v8": "detect.pt",
    "YOLO v5": "ppe.pt",
    "YOLO v3": "best.pt"
}

# Function to process uploaded image and perform object detection using the selected model
def process_image(img, selected_model):
    model_path = models[selected_model]
    model = YOLO(model_path)
    classNames = model.names

    results = model(img)

    detection_results = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Process each detected box
            cls = int(box.cls)
            current_class = classNames[cls]
            conf = box.conf.item()

            # Assign color based on class
            if current_class in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']:
                color = (0, 0, 255)  # Red for non-compliance
            elif current_class in ['Hardhat', 'Safety Vest', 'Mask']:
                color = (0, 255, 0)  # Green for compliance
            else:
                color = (255, 0, 0)  # Blue for other objects

            # Append detection result as dictionary
            detection_results.append({'class_name': current_class, 'confidence': f'{conf:.2f}'})

            # Draw bounding box on the image
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Annotate the image with class and confidence
            text = f'{current_class}: {conf:.2f}'
            cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Convert the annotated image to a base64 string
    _, buffer = cv2.imencode('.jpg', img)
    img_str = base64.b64encode(buffer).decode()
    return img_str, detection_results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html', models=models)

@app.route('/select_model', methods=['GET', 'POST'])
def select_model():
    if request.method == 'POST':
        selected_model = request.form.get('model')
        return render_template('upload.html', selected_model=selected_model, models=models)
    return render_template('select_model.html', models=models)


@app.route('/upload_file', methods=['POST'])
def upload_file():
    selected_model = request.form.get('model')
    if selected_model is None:
        return render_template('error.html', message='Please select a model')

    if 'file' not in request.files:
        return render_template('error.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('error.html', message='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Read the uploaded image
        img = cv2.imread(file_path)
        
        # Process the uploaded image using the selected model
        image_data, detection_results = process_image(img, selected_model)
        
        # Check if any of the detection results indicate non-compliance
        non_compliance_detected = any(result['class_name'] in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask'] for result in detection_results)

        if non_compliance_detected:
            # Construct the email message
            msg = EmailMessage()
            msg['Subject'] = 'ALERT'
            msg['From'] = 'ms4400073@gmail.com'
            msg['To'] = 'snehasebastian062023@gmail.com'
            
            # Filter the detection results to include only non-compliance classes
            non_compliance_results = [result for result in detection_results if result['class_name'] in ['NO-Hardhat', 'NO-Safety Vest', 'NO-Mask']]
            
            # Compose the detection results in the email body
            html_body = '<h1>Detection Results</h1><br>' + \
                '<table><tr><th>Class</th><th>Confidence Score</th></tr>' + \
                ''.join(f'<tr><td>{result["class_name"]}</td><td>{result["confidence"]}</td></tr>' for result in non_compliance_results) + \
                '</table>'
            
            # Set the HTML content directly
            msg.set_content(html_body, subtype='html')

            # Attach the annotated image to the email
            with open(file_path, 'rb') as f:
                img_data = f.read()
            msg.add_attachment(img_data, maintype='image', subtype='jpg', filename=filename)

            # Send the email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login('ms4400073@gmail.com', 'szsj zkvx wlbx jeqh')
                server.send_message(msg)
            
        # Render the result page with annotated image and detection results
        return render_template('result.html', image_data=image_data, detection_results=detection_results, model_used=selected_model)

    
    ...
@app.route('/detect_hard_hats', methods=['POST'])
def detect_hard_hats():
    if 'file' not in request.files:
        return render_template('error.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('error.html', message='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Read the uploaded image
        img = cv2.imread(file_path)

        # Load the hardhat detection model
        hardhat_model_path = "hardhat.pt"
        hardhat_model = YOLO(hardhat_model_path)
        hardhat_class_names = hardhat_model.names

        # Perform object detection using the hardhat model
        results = hardhat_model(img)

        detection_results = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls)
                current_class = hardhat_class_names[cls]
                conf = box.conf.item()

                # Assign color based on class
                color = (0, 255, 0)  # Green for hard hats

                # Append detection result as dictionary
                detection_results.append({'class_name': current_class, 'confidence': f'{conf:.2f}'})

                # Draw bounding box on the image
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # Annotate the image with class and confidence
                text = f'{current_class}: {conf:.2f}'
                cv2.putText(img, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Convert the annotated image to a base64 string
        _, buffer = cv2.imencode('.jpg', img)
        img_str = base64.b64encode(buffer).decode()

        # Render the result page with annotated image and detection results
        return render_template('result.html', image_data=img_str, detection_results=detection_results,model_used='YOLO v5 (Scratch Code)')

    ...

@app.route('/detect_all_models', methods=['POST'])
def detect_all_models():
    if 'file' not in request.files:
        return render_template('error.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('error.html', message='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Read the uploaded image
        img = cv2.imread(file_path)

        # Perform object detection using all models
        detection_results = {}
        for model_name, model_path in models.items():
            image_data, results = process_image(img.copy(), model_name)
            detection_results[model_name] = {'image_data': image_data, 'results': results}

        return render_template('all_models_result.html', detection_results=detection_results)



if __name__ == '__main__':
    app.run(debug=True)
