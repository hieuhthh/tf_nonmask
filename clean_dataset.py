import os
import shutil
import multiprocessing
import cv2

def clean_image(route, to_des, im_size):
    """
    using multiprocessing
    input:
        route to main directory and phrase ("train", "valid", "test")
        or just route to the directory that its subfolder are classes
    output:
        X_path: path to img
        Y_int: int label
        all_class: list of string class name
    """

    global task

    def task(route, all_class, list_cls, to_des, im_size):
        print('Start task')

        sign = route.split('/')[-1]

        for cl in list_cls:
            path2cl = os.path.join(route, cl)

            if len(os.listdir(path2cl)) < 1:
                continue

            des_class = os.path.join(to_des, sign + '_' + cl)

            try:
                os.mkdir(des_class)
            except:
                pass

            for imfile in os.listdir(path2cl):
                impath = os.path.join(path2cl, imfile)
                imsave = os.path.join(des_class, imfile)

                try:
                    img = cv2.imread(impath)
                    img = cv2.resize(img, (im_size, im_size))
                    cv2.imwrite(imsave, img)
                except:
                    print(impath)

        print('Finish')

    cpu_count = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(cpu_count)
    processes = []

    all_class = sorted(os.listdir(route))
    n_labels = len(all_class)
    n_per = int(n_labels // cpu_count + 1)

    for i in range(cpu_count):
        print(f'Start cpu {i}')

        start_pos = i * n_per
        end_pos = (i + 1) * n_per
        list_cls = all_class[start_pos:end_pos]
     
        p = pool.apply_async(task, args=(route,all_class,list_cls,to_des,im_size))
        processes.append(p)

        # task(route,all_class,list_cls)

    result = [p.get() for p in processes]

    pool.close()
    pool.join()

if __name__ == '__main__':
    from utils import *

    settings = get_settings()
    globals().update(settings)

    des = path_join(route, 'dataset')

    try:
        shutil.rmtree(des)
    except:
        pass

    try:
        os.mkdir(des)
    except:
        pass

    route = 'unzip/gnv_dataset'
    clean_image(route, des, im_size)

    route = 'unzip/VN-celeb'
    clean_image(route, des, im_size)

    route = 'unzip/glint360k_224'
    clean_image(route, des, im_size)