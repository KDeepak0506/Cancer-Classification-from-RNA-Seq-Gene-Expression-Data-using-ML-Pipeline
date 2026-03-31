from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / r'dataset/raw/TCGA-PANCAN-HiSeq-801x20531/data.csv'
LABEL_PATH = BASE_DIR / r'dataset/raw/TCGA-PANCAN-HiSeq-801x20531/labels.csv'

def Load_Data():
  feature = pd.read_csv(DATA_PATH,index_col=0)
  label = pd.read_csv(LABEL_PATH,index_col=0)

  # convert a indexed sample_id col into a normal column.
  feature.reset_index(inplace=True)
  label.reset_index(inplace=True)

  # change the column name to 'SampleID' for readability.
  feature.rename(columns={feature.columns[0]:'SampleID'},inplace=True)
  label.rename(columns={label.columns[0]:'SampleID'},inplace=True)

  # merging both as a single Dataset by 'SampleID'.
  df_merge = pd.merge(feature,label,on='SampleID')
  return df_merge