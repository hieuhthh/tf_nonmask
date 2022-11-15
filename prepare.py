import os
import shutil

ROUTE = '/home/lap14880/hieunmt/faceid/nonmask'
DES = ROUTE + '/dataset'

# try:
#     shutil.rmtree(DES)
# except:
#     pass

try:
    os.mkdir(DES)
except:
    pass

def move_folder(route, pad=None):
    if pad is None:
        pad = route.split('/')[-1]
    print(pad, len(os.listdir(route)))
    for filename in os.listdir(route): 
        from_folder = route + '/' + filename
        to_folder = DES + '/' + pad + filename

        if len(from_folder) < 1:
            print(from_folder)
            continue

        try:
            shutil.copytree(from_folder, to_folder)
        except:
            print('Already have', filename)

print('N class before:', len(os.listdir(DES)))

# route = '/home/lap14880/hieunmt/faceid/nonmask/download/msm1_part1'
# move_folder(route)

# route = '/home/lap14880/hieunmt/faceid/nonmask/download/content/content/Keras_insightface/ms1m-retinaface-t1_112x112_folders'
# move_folder(route)

# route = '/home/lap14880/hieunmt/faceid/nonmask/download/VN-celeb'
# move_folder(route)

# route = '/home/lap14880/hieunmt/faceid/nonmask/download/glint360k_224'
# move_folder(route)

# route = '/home/lap14880/hieunmt/faceid/nonmask/download/with_crop_client1706'
# move_folder(route)

route = '/home/lap14880/hieunmt/faceid/nonmask/download/msra'
move_folder(route)

print('N class:', len(os.listdir(DES)))

    