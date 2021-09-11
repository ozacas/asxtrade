#!/usr/bin/python3.8
import pymongo
import argparse
import json
import os
from bson import ObjectId
from utils import *

def fix_percentage_fields(quotations): 
    assert quotations is not None
    updates = {}
    for rec in quotations.find({ }):
        assert '_id' in rec
        id = rec.get('_id')
        updates[id] = {}
        for field_name in ['change_in_percent', 'previous_day_percentage_change']:
            updates[id][field_name] = fix_percentage(rec, field_name)
        #print("{}={}".format(id, updates[id]))

    print("About to update {} quotation records.".format(len(updates.keys())))
    for id, fields_to_update in updates.items():
        result = quotations.update_one({ '_id': ObjectId(id) }, { '$set': fields_to_update } )
        assert result is not None
        assert result.matched_count == 1
        assert result.modified_count < 2
    print("Run complete.")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Fetch and update data from ASX Research")
    args.add_argument('--config', help="Configuration file to use [config.json]", type=str, default="config.json")
    a = args.parse_args()

    config = {}
    with open(a.config, 'r') as fp:
        config = json.loads(fp.read())
    m = config.get('mongo')
    print(m)
    password = m.get('password')
    if password.startswith('$'):
        password = os.getenv(password[1:])
    mongo = pymongo.MongoClient(m.get('host'), m.get('port'), username=m.get('user'), password=password)
    db = mongo[m.get('db')]
    fix_percentage_fields(db.asx_prices)
    mongo.close()
    print("Run completed.")
    exit(0)
