import librosa


def getSamples(file_name, sample_rate):
    samples_test, sample_rate = librosa.load(file_name, res_type="kaiser_fast", sr=sample_rate)
    return samples_test

#generaly chosse first 13, by default n_mfcc=20
# https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b

def getMfcc(samples, sample_rate):
    mfcc = librosa.feature.mfcc(y=samples, sr=sample_rate, n_mfcc=13)
    return mfcc

def getTrimmed(samples):
    samples_trimmed, index = librosa.effects.trim(samples)
    return samples_trimmed

