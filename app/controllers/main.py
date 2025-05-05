from flask import Blueprint, render_template, request, current_app
import os
from werkzeug.utils import secure_filename
from app.models.processor import apply_selected_operator, apply_threshold

main_bp = Blueprint('main', __name__)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@main_bp.route('/')
def index():
    return render_template('index.html')

@main_bp.route('/edge-detection', methods=['GET', 'POST'])
def edge():
    if request.method == 'POST':
        image = request.files.get('image')
        operator = request.form.get('operator')

        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            upload_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
            image.save(upload_path)

            result_filename = f"{operator}_{filename}"
            result_path = os.path.join(current_app.config['RESULT_FOLDER'], result_filename)

            apply_selected_operator(upload_path, operator, result_filename)

            return render_template('edge.html', selected_operator=operator,
                original_image=f"uploads/{filename}",
                result_image=f"results/{result_filename}")

    return render_template('edge.html')

@main_bp.route('/threshold', methods=['GET', 'POST'])
def threshold():
    if request.method == 'POST':
        threshold_value = int(request.form.get('threshold'))
        image = request.files.get('image')
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            upload_folder = current_app.config['UPLOAD_FOLDER']
            upload_path = os.path.join(upload_folder, filename)
            image.save(upload_path)

            result_filename = f"threshold_{filename}"
            result_path = os.path.join(current_app.config['RESULT_FOLDER'], result_filename)

            apply_threshold(upload_path,threshold_value, result_path)

            return render_template('threshold.html', 
                original_image=f"uploads/{filename}",
                threshold_image=f"results/{result_filename}")
    
    return render_template('threshold.html')