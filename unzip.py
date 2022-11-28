import zipfile
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import time

from utils import *

settings = get_settings()
globals().update(settings)

from_dir = path_join(route, 'download')
des = path_join(route, 'unzip')
force_mkdir(des)

def fast_unzip(zip_path, out_path):
    start = time.time()
    with ZipFile(zip_path, 'r') as handle:
        with ThreadPoolExecutor(2) as exe:
            _ = [exe.submit(handle.extract, m, out_path) for m in handle.namelist()]
    print('Unzip', zip_path, 'Time:', time.time() - start)

filename = 'vnceleb'
zip_path = path_join(from_dir, filename + '.zip')
fast_unzip(zip_path, des)

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#     zip_ref.extractall(des)

# filename = 'gnv_dataset'
# zip_path = path_join(from_dir, filename + '.zip')
# fast_unzip(zip_path, des)

# zip_path = '/home/lap14880/face_bucket_huy/glint360k_224_copy.zip'
# fast_unzip(zip_path, des)

