# Imports
import numpy as np
import gradio as gr
import librosa
from joblib import load
from keras.models import load_model
from gradio.components import Textbox, Audio

# global variables
labels = {
    0: 'Angry',
    1: 'Disgusted',
    2: 'Fearful',
    3: 'Happy',
    4: 'Neutral',
    5: 'Sad',
    6: 'Suprised'
}

SAMPLE_RATE = 22050
SIGNAL_LENGTH = int(SAMPLE_RATE * 1.5) # 1.5 seconds
MFCC_NUM = 65

x_mean = -4.222186
x_std = 54.317802

# Preprocessing functions
def remove_silence(signal, threshold=0.005):
    """
    Remove silence at the beginning and at the end of the signal
    """
    i = j = 0
    for i in range(len(signal)):
        if np.abs(signal[i]) > threshold:
            break
    for j in range(len(signal)-1, 0, -1):
        if np.abs(signal[j]) > threshold:
            break
    return signal[i:j]

def signal_resize(signal, length=SIGNAL_LENGTH):
    """
    Cut the signal to the given length, or pad it with zeros if it is shorter
    """
    length = length
    if len(signal) > length:
        return signal[:length]
    else:
        return np.pad(signal, (0, max(0, length - len(signal))), "constant")

def get_percentage(num):
    """
    Get the percentage of the given number with 4 decimals
    """
    return str(int(num * 1000000)/10000) + ' %'


################################# Gradio Interface #################################

## Inputs & Outputs
speech_path = Audio(label="Speech audio", source="upload", type="filepath")
output1 = Textbox(label="Predicted Emotion")
output2 = Textbox(label="Angry Score")
output3 = Textbox(label="Disgusted Score")
output4 = Textbox(label="Fearful Score")
output5 = Textbox(label="Happy Score")
output6 = Textbox(label="Neutral Score")
output7 = Textbox(label="Sad Score")
output8 = Textbox(label="Suprised Score")



### Function to predict the emotion of the audio file
def predictor_svm(path):
    """
    Function that takes the path of the audio file and returns the predicted emotion
    """
    # read the audio file
    signal, sample_rate = librosa.load(path, res_type='kaiser_fast')

    # Load the model
    model = load('./Model/SVMmodel.joblib')
    scaler = load('./Model/std_scaler.bin')

    # Preprocess the signal
    signal = remove_silence(signal)
    signal = signal_resize(signal)
    mfcc_values = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=40).T
    mfccs_mean = np.mean(mfcc_values, axis=0) # media en los coeficientes
    data = scaler.transform(mfccs_mean.reshape(1, -1))

    # Predict the emotion
    prediction = model.predict(data)[0]
    print(prediction)

    return labels[prediction]

def predictor_nn(path):
    """
    Function that takes the path of the audio file and returns the predicted emotion
    """
    # read the audio file
    signal, sample_rate = librosa.load(path, res_type='kaiser_fast')

    # Load the model
    model = load_model('./Model/redneuronal.h5')

    # Preprocess the signal
    signal = remove_silence(signal)
    signal = signal_resize(signal)
    mfcc_values = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=128).T
    mfccs_mean = np.mean(mfcc_values, axis=0) # media en los coeficientes
    mfccs_std = np.std(mfcc_values, axis=0) # Desviación estandar en los coeficientes
    data = np.concatenate((mfccs_mean, mfccs_std), axis=0)
    item = data.reshape(-1, 256, 1)

    # Predict the emotion
    prediction = model.predict(item)[0]

    return [ labels[np.argmax(prediction)],
        get_percentage(prediction[0]), get_percentage(prediction[1]),
        get_percentage(prediction[2]), get_percentage(prediction[3]),
        get_percentage(prediction[4]), get_percentage(prediction[5]),
        get_percentage(prediction[6])
    ]

# Interface Functions
def predictor_conv(path):
    """
    Function that takes the path of the audio file and returns the predicted emotion
    """
    # read the audio file
    signal, sample_rate = librosa.load(path, res_type='kaiser_fast')

    # Load the model
    model = load_model('./Model/convolutional.h5')

    # Preprocess the signal
    signal = remove_silence(signal)
    signal = signal_resize(signal)
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=MFCC_NUM).T
    data = np.asarray(mfcc)
    item = data.reshape(-1, 65, 65, 1)
    norm = (item - x_mean) / x_std

    # Predict the emotion
    prediction = model.predict(norm)[0]

    return [ labels[np.argmax(prediction)],
        get_percentage(prediction[0]), get_percentage(prediction[1]),
        get_percentage(prediction[2]), get_percentage(prediction[3]),
        get_percentage(prediction[4]), get_percentage(prediction[5]),
        get_percentage(prediction[6])
    ]

## Interfaces

# Suport Vector Machine
svm_iface = gr.Interface(
    fn = predictor_svm, 
    inputs = [speech_path], 
    outputs = [output1], 
    live = False, 
    title = "Audio Speech Sentiment Classifier - Support Vector Machine",
    description = "Audio Speech Sentiment Classifier using Machine Learning techniques"
)

# Neural Network Interface
nn_iface = gr.Interface(
    fn = predictor_nn, 
    inputs = [speech_path], 
    outputs = [output1, output2, output3, output4, output5, output6, output7, output8], 
    live = False, 
    title = "Audio Speech Sentiment Classifier - Neural Network",
    description = "Audio Speech Sentiment Classifier using Machine Learning techniques"
)

# Convolutional Interface
conv_iface = gr.Interface(
    fn = predictor_conv, 
    inputs = [speech_path], 
    outputs = [output1, output2, output3, output4, output5, output6, output7, output8], 
    live = False, 
    title = "Audio Speech Sentiment Classifier - Convolutional Neural Network",
    description = "Audio Speech Sentiment Classifier using Machine Learning techniques"
)

demo = gr.TabbedInterface(
    [svm_iface, nn_iface, conv_iface], 
    ["Suport Vector Machine", "Deep Neuronal Network", "Convolutional Network"]
)

if __name__ == "__main__":
    demo.launch()