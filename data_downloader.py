import os, zipfile, pandas as pd, gdown

data_dir = os.path.join(os.getcwd(), "data")
os.makedirs(data_dir, exist_ok=True)

share_urls = {"images.zip": "https://drive.google.com/file/d/1JL1ELJl2lXXDSISqCKhCGLKPlydkJCtd/view?usp=sharing",
	      "timeseries.zip": "https://drive.google.com/file/d/1RL48z46AuJrqw4521CVuxEFaAuYEuQLB/view?usp=sharing"}

for folderN, share_url in share_urls.items():
	zip_path = os.path.join(data_dir, folderN)
	gdown.download(url=share_url, output=zip_path, quiet=False, fuzzy=True)

	os.makedirs(zip_path.replace(".zip", ""), exist_ok=True)
	with zipfile.ZipFile(zip_path, 'r') as z:
		z.extractall(zip_path.replace(".zip", ""))
