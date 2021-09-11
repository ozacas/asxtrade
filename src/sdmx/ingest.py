import pandasdmx as sdmx
import pandas as pd

def dump_dimension(dsd, dim_name: str, all:bool=False):
    if all:
        print(dsd.dimensions.components)
        print(dsd.attributes.components)
    e = dsd.dimensions.get(dim_name).local_representation.enumerated
    df = sdmx.to_pandas(e)
    print(df)

def dump_flows(data_source: sdmx.Request):
    flows = data_source.dataflow()
    df = sdmx.to_pandas(flows.dataflow)
    print(df)

def dump_structure(data_source: sdmx.Request, dataset_name: str, debug=False):
    print(f"Exploring {dataset_name}...")

    dataset_msg = data_source.dataflow(dataset_name)
    dataset = getattr(dataset_msg.dataflow, dataset_name)
    if debug:
        print(dataset)
    dsd = dataset.structure
    if debug:
        print(dsd)
    assert isinstance(dsd, sdmx.model.DataStructureDefinition)
    return dsd

def get_available_of_dimension(dsd: sdmx.model.DataStructureDefinition, dim_name: str):
    dim = dsd.dimensions.get(dim_name).local_representation.enumerated
    available_results = sdmx.to_pandas(dim)
    #print(type(available_results))
    assert isinstance(available_results, pd.Series)
    return available_results

def ecb():
    ecb = sdmx.Request('ECB', backend='sqlite', fast_save=True, expire_after=6000)
    dump_flows(ecb)
    dsd = dump_structure(ecb, 'YC')
    for dim_name in ['FREQ', 'REF_AREA', 'DATA_TYPE_FM']:
        results = get_available_of_dimension(dsd, dim_name)
        print(results)

    # external trade for australia since 2000
    dsd = dump_structure(ecb, 'TRD')
    print("*** External Trade")
    dump_dimension(dsd, 'REF_AREA', all=True)
    dump_dimension(dsd, 'TRD_PRODUCT')
    dump_dimension(dsd, 'TRD_FLOW')
    resp = ecb.data('TRD', key={'REF_AREA': 'I8'})
    df = resp.to_pandas()
    print(df)

if __name__ == '__main__':
   wb = sdmx.Request('WB', backend='sqlite', fast_save=True, expire_after=6600)
   dump_flows(wb)
   dsd = dump_structure(wb, 'DF_WITS_TradeStats_Tariff', debug=True)
   available_results = get_available_of_dimension(dsd, 'TRADESTATS')