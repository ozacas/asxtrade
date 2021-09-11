"""
Responsible for implementing download of datasets (CSV/Excel/TSV/Parquet)
on various pages for the user
"""
import tempfile
from django.http import HttpResponse, Http404
from django.contrib.auth.decorators import login_required
from app.models import (
    cached_all_stocks_cip,
    Timeframe,
    stocks_by_sector,
    validate_user,
    user_watchlist,
    all_etfs,
    all_sector_stocks,
    Sector,
    validate_stock,
    financial_metrics,
)
from app.data import make_kmeans_cluster_dataframe, make_pe_trends_eps_df, pe_trends_df


def save_dataframe_to_file(df, filename, output_format):
    assert output_format in ("csv", "excel", "tsv", "parquet")
    assert df is not None and len(df) > 0
    assert len(filename) > 0

    if output_format == "csv":
        df.to_csv(filename)
        return "text/csv"
    elif output_format == "excel":
        df.to_excel(filename, engine="xlsxwriter")
        return "application/vnd.ms-excel"
    elif output_format == "tsv":
        df.to_csv(filename, sep="\t")
        return "text/tab-separated-values"
    elif output_format == "parquet":
        df.to_parquet(filename)
        return "application/octet-stream"  # for now, but must be something better...
    else:
        raise ValueError("Unsupported format {}".format(output_format))


def get_dataset(dataset_wanted, request):
    assert (
        dataset_wanted in set(["market_sentiment", "eps-per-sector"])
        or dataset_wanted.startswith("kmeans-")
        or dataset_wanted.startswith("financial-metrics-")
    )

    if dataset_wanted == "market_sentiment":
        df = cached_all_stocks_cip(Timeframe())
        return df
    elif dataset_wanted == "kmeans-watchlist":
        timeframe = Timeframe(past_n_days=300)
        _, _, _, _, df = make_kmeans_cluster_dataframe(
            timeframe, 7, user_watchlist(request.user)
        )
        return df
    elif dataset_wanted == "kmeans-etfs":
        timeframe = Timeframe(past_n_days=300)
        _, _, _, _, df = make_kmeans_cluster_dataframe(timeframe, 7, all_etfs())
        return df
    elif dataset_wanted.startswith("kmeans-sector-"):
        sector_id = int(dataset_wanted[14:])
        sector = Sector.objects.get(sector_id=sector_id)
        if sector is None or sector.sector_name is None:
            raise Http404("No stocks associated with sector")
        asx_codes = all_sector_stocks(sector.sector_name)
        timeframe = Timeframe(past_n_days=300)
        _, _, _, _, df = make_kmeans_cluster_dataframe(timeframe, 7, asx_codes)
        return df
    elif dataset_wanted.startswith("financial-metrics-"):
        stock = dataset_wanted[len("financial-metrics-") :]
        validate_stock(stock)
        df = financial_metrics(stock)
        if df is not None:
            # excel doesnt support timezones, so we remove it first
            colnames = [d.strftime("%Y-%m-%d") for d in df.columns]
            df.columns = colnames
            # FALLTHRU
        return df
    elif dataset_wanted == "eps-per-sector":
        df, _ = pe_trends_df(Timeframe(past_n_days=180))
        df = make_pe_trends_eps_df(df, stocks_by_sector())
        df = df.set_index("asx_code", drop=True)
        return df
    else:
        raise ValueError("Unsupported dataset {}".format(dataset_wanted))


@login_required
def download_data(request, dataset=None, output_format="csv"):
    validate_user(request.user)

    with tempfile.NamedTemporaryFile() as fh:
        df = get_dataset(dataset, request)
        if df is None or len(df) < 1:
            raise Http404("No such dataset: {}".format(dataset))
        content_type = save_dataframe_to_file(df, fh.name, output_format)
        fh.seek(0)
        response = HttpResponse(fh.read(), content_type=content_type)
        extension_by_format = {
            "csv": "csv",
            "excel": "xlsx",
            "parquet": "pq",
            "tsv": "tsv",
        }
        extension = extension_by_format[output_format]
        response["Content-Disposition"] = "inline; filename=temp.{}".format(extension)
        return response
