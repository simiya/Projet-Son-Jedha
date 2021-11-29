import streamlit as st
import os
import matplotlib.pyplot as plt
import wordcloud
from wordcloud import WordCloud
from PIL import Image


st.write("""
# Hacensor
Déposez vos fichiers audio ! Si c'est trash on censure pour vous :)
""")

st.image("Logo.png", width=150)



uploaded_file = st.file_uploader("Importer un fichier", type=["wav", "mp3", "m4a"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.audio(uploaded_file, format='audio')

    with open(os.path.join("Import_Audio",uploaded_file.name),"wb") as f:
        f.write(uploaded_file.getbuffer())

    #Lancement du script de Margot
    exec(open("Script_Margot.py").read())

    #Lancement du script de Pierre et Youssef
    import Script_PiYo
    from Script_PiYo import main
    output_final, Message = main()

    #Import Script Mala
    from Script_Mala import GetPathAudio
    from Script_Mala import Transcription

    Path_Audio = GetPathAudio()
    #Lancement de la transcription
    Text, Start, End, Transcription_audio_to_text = Transcription()


    if output_final >= 1:

        if output_final == 1 :
            image = Image.open('Option1.jpg')
            Message2 = "C'est que vous êtes un peu vulgaire vous ! On va nettoyer tout ça  !"

        elif output_final == 2 :
            image = Image.open('Option2.jpg')
            Message2 = "Attention ! Ton langage ne me plaît pas trop ! Je vais nettoyer tout ça"

        elif output_final == 3 :
            image = Image.open('Option3.jpg')
            Message2 = "Tu parles beaucoup trop mal, tu l'auras bien cherché ! Tu as 5 secondes pour télécharger le fichier propre ou ça va mal se passer !"

        st.image(image)

        st.write(Message2)

        st.download_button("Télécharger", Path_Audio)
        st.audio(Path_Audio, format='audio')


        #Lancement du script de Malamine
        #exec(open("Script_Mala.py").read())

        st.write("""------------------------- Transcription de votre audio -------------------------""")
        st.write(Transcription_audio_to_text)


        # generate a word cloud excluding stopwords from the text file

        st.write("Nuage de Mots")
        st.set_option('deprecation.showPyplotGlobalUse', False)

        wd = wordcloud.WordCloud(background_color="white", contour_width=1, stopwords=[], max_words=50)
        # Generate wordcloud 
        cloud = wd.generate(Transcription_audio_to_text)

        # Show plot
        plt.imshow(cloud)
        plt.axis("off")
        plt.show()
        st.pyplot()

    else:
        st.write()

else: 
    st.error('Veuillez déposer un fichier audio.')





"""
#Remove all files from import directory at the end of the process
dir = 'Import_Audio'
for f in os.listdir(dir):
    os.remove(os.path.join(dir, f))"""