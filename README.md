# Projet-Son-Jedha
Projet Subject:  mood dection on speech audio

TODO: decide the final subject , deadline next wendesday (data, detailed project ready)
if vulgarity, must find the exsiting data
the total project time is 2 weeks

1. sources:

https://scholar.google.fr/scholar?q=detection+of+feeling+audio&hl=fr&as_sdt=0&as_vis=1&oi=scholart

https://www.google.com/intl/fr/chrome/demos/speech.html



1.1 Audio

1.1.1 There is the link of data used in this projet.
https://towardsdatascience.com/classifying-emotions-using-audio-recordings-and-python-434e748a95eb

1.1.2 It uses transfer learning
https://towardsdatascience.com/self-supervised-voice-emotion-recognition-using-transfer-learning-d21ef7750a10

1.2 music
1.2.1
https://blog.clairvoyantsoft.com/music-genre-classification-using-cnn-ef9461553726 (on peut avoir le genre aussi)
1.2.2
https://towardsdatascience.com/predicting-the-music-mood-of-a-song-with-deep-learning-c3ac2b45229e

1.3. speech 
1.3.1
https://discourse.mozilla.org/c/deepspeech/247


Steps

1. find data
  1.1 RAVDESS;  TESS; SAVEE in 1.1.1
2. numerise audio file 
3. find pre-proceeded models
https://hub.tensorflow.google.cn/s?module-type=audio-embedding
4. training
5. conclusion
6. on streaming treatement
very interesting document
https://www.frontiersin.org/articles/10.3389/fcomp.2020.00014/full

8. extra: speech to text, bad word detection

- Spleeter : développé par Deezer, permet de séparer voix/instruments en deux fichiers distincts    ---------- preprocessing 
- DeepSpeech : développé par Mozilla, speech to text (un peu biaisé car fonctionne très bien sur des sons avec peu de noise, et mieux avec des voix américaines masculines, donc à utiliser après Spleeter)    -------- reutilise API pour traiter les textes après
