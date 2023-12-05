from flask import Flask, render_template, request, send_file, url_for
import os
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['PROCESSED_FOLDER'] = 'static/'

# Load YOLO model (adjust the path as needed)
model = YOLO('best_v3.pt')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    confidence = float(request.form.get('confidence', 0.5))
    overlap = float(request.form.get('overlap', 0.3))
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        processed_image_filename, html_filename = process_image(file_path, confidence, overlap)
        return render_template('download.html', 
                               image_filename=processed_image_filename, 
                               html_filename=html_filename)

def draw_box(image, box, label):
    # Draw a rectangle and label on the image
    draw = ImageDraw.Draw(image)
    draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=2)
    draw.text((box[0] + 10, box[1] + 10), label, fill="red")

def process_image(image_path, confidence, overlap):
    # Load image with PIL
    img = Image.open(image_path)

    # Perform inference
    results = model.predict(img, conf=confidence, iou=overlap, imgsz=(640, 640))

    # Initialize an array to store details of detected objects
    objects = []

    # Draw boxes and labels on the image and store details
    for result in results:
        boxes = result.boxes
        for box, class_id in zip(boxes.xyxy, boxes.cls):
            # Draw each box on the image
            draw_box(img, box, result.names[int(class_id)])
            # Store object details
            x_min, y_min, x_max, y_max = box
            class_name = result.names[int(class_id)]
            objects.append({
                "x_min": x_min.item(),
                "y_min": y_min.item(),
                "width": (x_max - x_min).item(),
                "height": (y_max - y_min).item(),
                "class": class_name
            })


    # Save the processed image
    processed_image_filename = 'processed_' + os.path.basename(image_path)
    processed_image_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_image_filename)
    img.save(processed_image_path)

    # Generate and save HTML file with detected objects
    html_filename = 'new_test_' + os.path.splitext(os.path.basename(image_path))[0] + '.html'
    html_file_path = os.path.join(app.config['PROCESSED_FOLDER'], html_filename)
    with open(html_file_path, 'w') as f:
    # Write the start of the HTML and CSS
                f.write("""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Document</title>
                <style>

                    .detected-object {
                        position: absolute;
                        border: 1px solid red;
                    }
                """)
                
                # Write CSS for each object
                for i, ob in enumerate(objects):
                    f.write(f"""
                    #{ob['class']}{i} {{
                        left: {int(ob['x_min'])}px;
                        top: {int(ob['y_min'])}px;
                        width: {int(ob['width'])}px;
                        height: {int(ob['height'])}px;
                    }}
                    """)
                
                # End of CSS and start of body
                f.write("""
                </style>
            </head>
            <body>
                """)

                # Write HTML elements for each object
                for i, ob in enumerate(objects):
                    if ob['class'] == 'image':
                        f.write(f'<img id="{ob["class"]}{i}" class="detected-object" src="default.png" alt="Default Image">\n')
                    elif ob['class'] == 'text':
                        f.write(f'<div id="{ob["class"]}{i}" class="detected-object">Some random text</div>\n')
                    elif ob['class'] == 'button':
                        f.write(f'<button id="{ob["class"]}{i}" class="detected-object">Button</button>\n')
                    elif ob['class'] == 'header':
                        # style="background-color: #2196F3; padding: 15px 20px; overflow: hidden;">
                        f.write(f"""<div id="{ob["class"]}{i}" style = "background-color: #2196F3;" class="detected-object">
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">Home</a>
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">About</a>
                                    <a href="#" style="float: left; color: #f2f2f2; text-decoration: none; font-size: 2.5rem; padding: 14px 16px;">Contact</a>
                                </div>
                            </header>\n""")
                    elif ob['class'] == 'footer':
                        f.write(f'<footer id="{ob["class"]}{i}" class="detected-object">Footer content</footer>\n')
                    elif ob['class'] == 'card':
                        f.write(f'<div id="{ob["class"]}{i}" class="detected-object"><img id="{ob["class"]}{i}" src="default.png" alt="Default Image"><h3 class="detected-object">Text for card</h3></div>\n')
                    elif ob['class'] == 'search_bar':
                        f.write(f"""
                        <div id="{ob["class"]}{i}" class="detected-object" >
                            <input id="{ob["class"]}{i} type="text" placeholder="Search">
                            <buttoni d="{ob["class"]}{i} >Search</button>
                        </div>
                        """)
                
                # End of HTML
                f.write("""
            </body>
            </html>
            """)


    return processed_image_filename, html_filename

@app.route('/download/<filename>')
def download_file(filename):
    file_path = os.path.join(os.getcwd(), app.config['PROCESSED_FOLDER'], filename)
    return send_file(file_path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
