from fileinput import filename
import os
import zipfile

def zip_files(directory, dest_name):
    zf = zipfile.ZipFile("{}.zip".format(dest_name), "w", zipfile.ZIP_DEFLATED)
    for files in os.listdir(directory):
        absfile = os.path.abspath(os.path.join(directory, files))
        arcname = absfile[absfile.rfind("/")+1:]
        print(absfile, arcname)
        zf.write(absfile, arcname)
    zf.close()

def unzip_files(filename, destination="") -> str:
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(destination)
