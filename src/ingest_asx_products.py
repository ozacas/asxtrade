#!/usr/bin/python3

import pandas as pd
import requests
import re
import pymongo
import argparse
from utils import read_config


def find_asx_code(df: pd.DataFrame) -> int:
   idx = 0
   for i, val in df.iloc[:,0].items():
      idx += 1
      if not isinstance(val, str):
          continue 
      if re.match(r'^ASX\s*Code', val):
          return idx-1
   return -1

def clean_sheet(sheet_idx:int, df:pd.DataFrame) -> pd.DataFrame:
   assert sheet_idx >= 0
   assert df is not None and len(df) > 0
   if sheet_idx in [0, 1, 5]: # skip rows before 'ASX Code' in first col
      row_idx = find_asx_code(df)
      assert row_idx > 0 
      #print(f"start row={row_idx} all columns for sheet {sheet_idx}")
      df = df.iloc[row_idx:,:]
      df = df.reset_index(drop=True)
      df.columns = df.iloc[0]
      df = df.iloc[1:,:]
      # drop rows with missing asx code or type 
      df = df.dropna(subset=(df.columns[0], df.columns[1]))
      return df
   return df

# VERY IMPORTANT: as it ensures nice clean database fields
def cleanup_column_heading(s: str) -> str:   
   if isinstance(s, str):
      s = s.replace('\n', '')
      s = s.replace('*', '')
      s = s.replace('#', '')
      s = s.replace('  ', ' ')
      s = s.replace('.', '-') # mongo rejects fields with . in them
      s = s.replace('\'', '-')
      s = s.replace('A$', 'AUD') # and dollar signs too... well ok mongo 5.0 is a little more flexible
      s = s.replace('$', 'AUD')
      s = s.replace('_', '-')
      s = s.strip()
   return s

def main(out_filename:str, db, template_url:str, month_abbrev:str="jan", year:int=2022):
   url = template_url.format(month_abbrev=month_abbrev, year=year)
   print(url)
   xl_resp = requests.get(url)
   assert xl_resp.status_code == 200
   idx = 0
   bytes=xl_resp.content
   if out_filename:
      with open(out_filename, 'w') as fp:
         fp.write(xl_resp.text)
   while idx >= 0:
      try:
         df = clean_sheet(idx, pd.read_excel(bytes, sheet_name=idx))
         #print(df.iloc[0,:])
         df.columns = [cleanup_column_heading(c) for c in df.columns]
         print(df.columns)
         for idx, row in df.iterrows():
            d = dict(row.dropna())
            d["month"] = month_abbrev
            d["year"] = year
            d['asx_code'] = row['ASX Code']
            db.asx_investment_products.update_one(
                {"asx_code": d['asx_code'], "month": month_abbrev, "year": year}, {"$set": d}, upsert=True
            )
            #print(row)
         idx += 1
      except Exception as e:
         if not isinstance(e, ValueError):
            raise e
         idx = -1


if __name__ == "__main__":
   args = argparse.ArgumentParser(
      description="Read Excel ASX investment products spreadsheet and load into database"
   )
   args.add_argument(
      "--config",
      help="Configuration file to use [config.json]",
      type=str,
      default="config.json",
   )
   args.add_argument("--debug", help="Debug spreadsheet contents", action="store_true")
   args.add_argument("--out", help="Filename to save raw XLS contents to [None]", type=str, default=None)
   ap = args.parse_args()
   config, password = read_config(ap.config)
   m = config.get("mongo")
   mongo = pymongo.MongoClient(
        m.get("host"), m.get("port"), username=m.get("user"), password=password
   )
   db = mongo[m.get("db")]
   main(ap.out, db, template_url=config.get("asx_investment_products_template_url"))
   exit(0)
