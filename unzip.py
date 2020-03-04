import pandas as pd
import subprocess
input_files = pd.read_csv('batch-2-files/down_sampled_positive_data.csv')['Labels'].tolist()
for f in input_files:
	cmd = "unzip -n /brain-hemorrag-v2/rsna-intracranial-hemorrhage-detection.zip rsna-intracranial-hemorrhage-detection/stage_2_train/{0}.dcm /brain-hemorrag-v2".format(f)
	subprocess.call(cmd.split())

