from django.contrib import admin
from app.models import (
    Quotation,
    Security,
    CompanyDetails,
    VirtualPurchase,
    Watchlist,
    Sector,
    MarketQuoteCache
)
from app.paginator import NoCountPaginator

@admin.register(Quotation)
class QuoteAdmin(admin.ModelAdmin):
    #date_hierarchy = 'year_high_date'
    list_display = ('_id', 'asx_code', 'last_price', 'last_trade_date', 
                    'annual_dividend_yield', 'eps', 'change_in_percent',
                    'volume')
    list_filter = ('suspended', 'year_high_date', 'year_low_date')
    paginator = NoCountPaginator
    show_full_result_count = False

@admin.register(Security)
class SecurityAdmin(admin.ModelAdmin):
    date_hierarchy = 'last_updated'
    list_display = ('_id', 'asx_code', 'asx_isin_code', 'company_name', 'security_name', )


@admin.register(VirtualPurchase)
class VirtualPurchaseAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'asx_code', 'buy_date', 'amount', 'n', 'price_at_buy_date')

@admin.register(CompanyDetails)
class CompanyDetailsAdmin(admin.ModelAdmin):
    list_display = ('_id', 'asx_code', 'name_full', 'phone_number', 
                    'principal_activities', 'web_address', )
    exclude = ('indices', 'products', 'latest_annual_reports')
    list_filter = ('recent_announcement', 'suspended', 'sector_name')
    # bid_price = model.FloatField()
    # offer_price = model.FloatField()
    # previous_close_price = model.FloatField()
    # average_daily_volume = model.IntegerField()
    # number_of_shares = model.IntegerField()
    # primary_share_code = model.TextField()
    # principal_activities = model.TextField()

@admin.register(Watchlist)
class WatchlistAdmin(admin.ModelAdmin):
    #date_hierarchy = 'when'
    list_display = ('id', 'user', 'asx_code',)

@admin.register(Sector)
class SectorAdmin(admin.ModelAdmin):
    list_display = ('id', 'sector_name', 'sector_id',)

@admin.register(MarketQuoteCache)
class MarketQuoteCacheAdmin(admin.ModelAdmin):
    exclude = ('dataframe',)
    list_display = ('tag', 'n_days', 'n_stocks', '_id', 'sha256')
    list_filter = ('field', 'last_updated',)
