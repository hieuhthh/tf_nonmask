import gdown

from utils import *

settings = get_settings()
globals().update(settings)

des = path_join(route, 'download')
mkdir(des)

url = "https://drive.google.com/file/d/1VCNkiFvNhxVTDklMkT3GaVwX0C624RE4/view?usp=sharing"
output =  f"{des}/vnceleb.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1-VYjrPIoVWkE7uwsNO6exayh7yhpPo0n/view?usp=share_link"
output =  f"{des}/gnv_dataset.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)