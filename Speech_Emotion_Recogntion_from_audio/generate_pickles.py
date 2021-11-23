import pandas as pd
import feature_extraction as fe
import numpy as np

def generate_picke(sampling_rate):
    df_raw = pd.read_csv('RAVDESS_speech.csv')
    df_raw.head()
    df = df_raw
    df['filePath'][0]
    # here we generate the MFCC paramaters from audio files
    #df['mfcc'] = [ feature_extraction.getMfcc(feature_extraction.getSamples(x),8000) for x in df['filePath']]
    df['mfcc_trimmed'] = [ fe.getMfcc( \
                fe.getTrimmed( \
                fe.getSamples(x,sampling_rate)), sampling_rate) for x in df['filePath']]
    type(df.mfcc_trimmed)
    with open('mfcc_trimmed_'+ str(sampling_rate) + '.npy', 'wb') as f:
        np.save(f, df.mfcc_trimmed)

generate_picke(22050)