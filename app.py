from flask import Flask, render_template, request, redirect, url_for
import os
from demucs.audio import AudioFile
import torch
from demucs import pretrained
from demucs.audio import AudioFile
from flask import send_from_directory
import soundfile as sf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        process_mp3(filename)
        return redirect(url_for('index'))
    

@app.route('/results')
def show_results():
    return render_template('results.html')



def process_mp3(file_path):
    # Load the pretrained Demucs model
    model = pretrained.load_pretrained('demucs')
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess the audio file
    with AudioFile(file_path) as mp3_file:
        waveform = mp3_file.read(resample=44100, channels_first=True, dtype='float32')

    # Ensure our waveform is on the GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    waveform = waveform.to(device)

    # Use the model to separate sources (vocals, drums, bass, and other)
    with torch.no_grad():
        sources = model(waveform)
    
    # Let's assume you just want to save vocals and instrumentals
    # vocals = sources[0]  # Index for vocals can change based on the model specifics
    # save_wav("vocals_output.wav", vocals.cpu(), 44100)

    # Combine all other sources to get the instrumentals
    # instrumentals = sum(sources[i] for i in range(1, len(sources)))
    # save_wav("instrumentals_output.wav", instrumentals.cpu(), 44100)
    

    with torch.no_grad():
        sources = model(waveform)

    # Now, save each source separately
    # source_names = ['vocals', 'drums', 'bass', 'other']
    # for index, source in enumerate(sources):
    #     save_wav(os.path.join(app.config['UPLOAD_FOLDER'], f"{source_names[index]}_output.wav"), source.cpu(), 44100)

     
    

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
