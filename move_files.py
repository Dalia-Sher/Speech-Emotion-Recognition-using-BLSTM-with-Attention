import shutil
import glob


new_path = "/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP_flat/ALL/"

for file in glob.glob("/home/dsi/shermad1/Emotion_Recognition/Data/IEMOCAP/IEMOCAP_full_release/*/sentences/wav/*/*.wav"):
    print(file)
    shutil.move(file, new_path)