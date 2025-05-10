from flask import Flask, request, render_template, redirect, url_for
import os
import torch
import numpy as np
import time
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from werkzeug.utils import secure_filename
from newrealtime import build_single_image_loader
from utils.utils import save_predict
from IRDPnet import IRDPNet
# Flask app setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/segmented_output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER
# checkpoint\cityscapes\IRDPnetbs4gpu1_train\model_500.pth
model =IRDPNet()
model = model.cuda()  # using GPU for inference
cudnn.benchmark = True
checkpoint_path = "checkpoint/cityscapes/IRDPnetbs4gpu1_train/model_500.pth"
checkpoint = torch.load(checkpoint_path, weights_only=True)
model.load_state_dict(checkpoint['model'])
model.eval()

def resize_image(image_path, output_size=(1024, 2048)):
    """Resize image to the desired output size and save it."""
    import cv2
    image = cv2.imread(image_path)
    resized_image = cv2.resize(image, output_size)
    resized_path = os.path.join(app.config['UPLOAD_FOLDER'], 'resized_' + os.path.basename(image_path))
    cv2.imwrite(resized_path, resized_image)
    return resized_path

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Save uploaded image
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            # Preprocess the image using CityscapesSingleImageProcessor
            input_image, size, image_name = build_single_image_loader(filepath)
            input_tensor = torch.from_numpy(input_image).float().unsqueeze(0)  # Add batch dimension

            # Perform prediction
            input_var = Variable(input_tensor).cuda()
            with torch.no_grad():
                start_time = time.time()
                output = model(input_var)
                torch.cuda.synchronize()
                time_taken = time.time() - start_time

                print(f"Prediction time: {time_taken:.2f} seconds")

                output = output.cpu().data[0].numpy()
                output = output.transpose(1, 2, 0)
                segmented_output = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)

            # Save the segmented output in static/segmented_output folder
            save_predict(segmented_output, None, image_name, "cityscapes", app.config['RESULT_FOLDER'],
                         output_grey=False, output_color=True, gt_color=False)

            segmented_image_path = f"static/segmented_output/{image_name}_color.png"
            print(f"Segmented image saved at: {segmented_image_path}")
            return redirect(url_for('result', original_image=filename, segmented_image=f"{image_name}_color.png"))

    return render_template('index.html')  # Render upload form

@app.route('/result')
def result():
    original_image = request.args.get('original_image')
    segmented_image = request.args.get('segmented_image')
    return render_template('result.html', original_image=original_image, segmented_image=segmented_image)

if __name__ == '__main__':
    app.run(debug=True)
