import os


folder_path='./data/images/'

for filename in os.listdir(folder_path):
    filename_new=filename[:-4].strip()+'.jpg'
    print(filename)
    src = folder_path + filename
    dst = folder_path + filename_new

    os.rename(src, dst)