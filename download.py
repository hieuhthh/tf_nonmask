import gdown
import os

ROUTE = '/home/lap14880/hieunmt/faceid/nonmask'
DES = ROUTE + '/download'

try:
    os.mkdir(DES)
except:
    pass

# url = "https://drive.google.com/file/d/1Iy5U5npDJDZBXtNMHl28jasr3-gr37Qu/view?usp=sharing"
# output = f"{DES}/with_crop_client1706.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1B2SS-fESvmK5fxJtcZRLQnUUSFJVV6WT/view?usp=sharing"
# output = f"{DES}/ms1m_part1.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1FmRKXvdav_a93bGpNMEY2DqdD34Zv5iR/view?usp=sharing"
# # url = "https://drive.google.com/file/d/1FmRKXvdav_a93bGpNMEY2DqdD34Zv5iR/view?usp=sharing"
# output =  f"{DES}/ms1m_part2.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)

# url = "https://drive.google.com/file/d/1VCNkiFvNhxVTDklMkT3GaVwX0C624RE4/view?usp=sharing"
# output =  f"{DES}/vnceleb.zip"
# gdown.download(url=url, output=output, quiet=False, fuzzy=True)