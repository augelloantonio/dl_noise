# ECG Noise recognizer model

Around 10 hours of ECG were manually labelled to differenziate Normal ECG signal and Artifact ECG signal. 

1250 samples of ECG for each raw where selected, rappresenting 10 seconds of ECG (250hz of sampling rate").
The column 1251 rappresents the annotated signal as:

- 0.0 = normal signal
- 1.0 = noise signal

## How to run

First need to install dependencies:
- pandas
- numpy
- tensorflow
- matplot

You can run the following comand to install all necessary libraries from the requirements file:
'''
pip install -r requirements.txt
'''

To run the program use the comand:

'''
python train.py
'''

This will produce also a .h5 model file.
