import os

directory = "./"

for filename in os.listdir(directory):
    if "_Jonathan" in filename:
        new_name = filename.replace("_Jonathan", "")
        os.rename(os.path.join(directory, filename), os.path.join(directory, new_name))

print("Renaming completed.")