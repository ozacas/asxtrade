"""
Responsible for ensuring all imports as required by urls.py are done
"""
from app.views.core import (
    show_all_stocks,
    show_etfs,
    png,
)
from app.views.search import (
    mover_search,
    company_search,
    dividend_search,
    market_cap_search,
    show_recent_sector,
    financial_metric_search,
    momentum_change_search,
)
from app.views.download import download_data
from app.views.optimise import (
    show_watchlist_outliers,
    show_sector_outliers,
    optimised_watchlist_view,
    optimised_etf_view,
    optimised_sector_view,
)
from app.views.watchlist import show_watched, toggle_watched, cluster_stocks_view
from app.views.stock import (
    show_stock,
    show_trends,
    show_total_earnings,
    show_purchase_performance,
    show_financial_metrics,
)
from app.views.virtual_purchases import (
    buy_virtual_stock,
    edit_virtual_stock,
    delete_virtual_stock,
)
from app.views.market import market_sentiment, show_pe_trends
