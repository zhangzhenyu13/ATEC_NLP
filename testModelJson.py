import os

model_dir="./models/"
files=os.listdir(model_dir)

for file in files:
    os.remove(model_dir+file)
