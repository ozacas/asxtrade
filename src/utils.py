import json
import os
import io
import hashlib
from datetime import datetime
import pytz
from bson.binary import Binary
import pandas as pd
import platform
import tempfile


def is_error(quotation: dict):
    assert quotation is not None
    if "error_code" not in quotation:
        return False
    return len(quotation.get("error_code")) == 0


def is_suspended(quotation: dict):
    assert quotation is not None
    if "suspended" not in quotation:
        return False
    return quotation.get("suspended", False) != False


def fix_percentage(quotation: dict, field_name: str):
    assert quotation is not None
    assert len(field_name) > 0
    if not field_name in quotation:
        return 0.0
    # handle % at end and ',' as the thousands separator
    field_value = quotation.get(field_name)
    if isinstance(field_value, str):
        val = field_value.replace(",", "").rstrip("%")
        pc = (
            float(val)
            if not any([is_error(quotation), is_suspended(quotation)])
            else 0.0
        )
        del quotation[field_name]
        assert field_name not in quotation
        quotation[field_name] = pc
        return pc
    else:
        quotation[field_name] = field_value  # assume already converted ie. float
        return field_value


def read_config(filename, verbose=True):
    """
    Read config.json (as specified by command line args) and return the password and
    configuration as a tuple
    """
    assert isinstance(filename, str)

    config = {}
    with open(filename, "r") as fp:
        config = json.loads(fp.read())
        m = config.get("mongo")
        if verbose:
            print(m)
        password = m.get("password")
        if password.startswith("$"):
            password = os.getenv(password[1:])
        return config, password


def now():
    return datetime.utcnow().replace(tzinfo=pytz.UTC)


def save_dataframe(
    mongo_collection, i: dict, df: pd.DataFrame, tag: str, scope: str
) -> None:
    assert mongo_collection is not None
    assert len(scope) > 0
    assert len(tag) > 0
    assert len(df) > 0
    if scope == "worldbank":
        dataframe_id = i["wb_id"]
    elif scope == "abs":
        dataframe_id = i["field"]
    elif scope == "ECB":
        dataframe_id = tag  # ie. URL fetched
    else:
        raise NotImplementedError()

    with io.BytesIO() as fp:
        # NB: if this fails it may be because you are using fastparquet
        # which doesnt (yet) support BytesIO. Use eg. pyarrow
        df.to_parquet(fp, compression="gzip", index=True)
        fp.seek(0)
        byte_content = fp.read()
        size = len(byte_content)
        sha256_hash = hashlib.sha256(byte_content).hexdigest()
        mongo_collection.update_one(
            {"tag": tag, "scope": scope},
            {
                "$set": {
                    "tag": tag,
                    "status": "CACHED",
                    "last_updated": now(),
                    "field": dataframe_id,
                    "market": "",
                    "scope": scope,
                    "n_days": 0,
                    "n_stocks": 0,
                    "dataframe_format": "parquet",
                    "size_in_bytes": size,
                    "sha256": sha256_hash,
                    "dataframe": Binary(byte_content),  # NB: always parquet format
                }
            },
            upsert=True,
        )
        print(f"{tag} == {size} bytes (sha256 hexdigest {sha256_hash})")


def get_tempfile():
    """
    As suggested in PR#7, some systems dont permit use of a named file when delete == True
    """
    tempfile_kwargs = {}
    if platform.system() == "Windows":
        tempfile_kwargs.update({"delete": False})
    return tempfile.NamedTemporaryFile(**tempfile_kwargs)
