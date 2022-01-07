from django import forms
from django.core.exceptions import ValidationError
from app.models import CompanyFinancialMetric, all_sector_stocks


def is_not_blank(value):
    if value is None or len(value) < 1 or len(value.strip()) < 1:
        raise ValidationError("Invalid value - cannot be blank")


def is_valid_sector(value):
    assert value is not None
    return len(all_sector_stocks(value)) > 0


class DividendSearchForm(forms.Form):
    min_yield = forms.FloatField(required=False, min_value=0.0, initial=4.0)
    max_yield = forms.FloatField(
        required=False, min_value=0.0, max_value=1000.0, initial=100.0
    )
    min_pe = forms.FloatField(
        required=False, min_value=0.0, initial=0.0, label="Min P/E"
    )
    max_pe = forms.FloatField(
        required=False, max_value=1000.0, initial=30.0, label="Max P/E"
    )
    min_eps_aud = forms.FloatField(
        required=False, min_value=-1000.0, initial=0.01, label="Min EPS ($AUD)"
    )


class CompanySearchForm(forms.Form):
    SECTOR_CHOICES = (
        ("Class Pend", "Class Pend"),
        ("Communication Services", "Communication Services"),
        ("Consumer Discretionary", "Consumer Discretionary"),
        ("Consumer Staples", "Consumer Staples"),
        ("Energy", "Energy"),
        ("Financials", "Financials"),
        ("Health Care", "Health Care"),
        ("Industrials", "Industrials"),
        ("Information Technology", "Information Technology"),
        ("Materials", "Materials"),
        ("Metals & Mining", "Metals & Mining"),
        ("Not Applic", "Not Applic"),
        ("Real Estate", "Real Estate"),
        ("Utilities", "Utilities"),
    )
    sector_enabled = forms.BooleanField(label="Match companies within chosen sectors?", required=False, initial=False)
    sector = forms.ChoiceField(
        choices=SECTOR_CHOICES,
        required=False,
        initial=SECTOR_CHOICES[1][0],
        validators=[is_not_blank, is_valid_sector],
    )
    name = forms.CharField(required=False)
    activity = forms.CharField(required=False)
    report_top_n = forms.IntegerField(required=False, min_value=10, max_value=2000)
    report_bottom_n = forms.IntegerField(required=False, min_value=10, max_value=2000)


class MoverSearchForm(forms.Form):
    METRIC_CHOICES = (
        ("annual_dividend_yield", "Change in dividend yield (%)"),
        ("eps", "Earnings per share (cents AUD)"),
        ("dividend_growth", "Growth in DY (%) above threshold"),
        ("eps_growth", "Growth in EPS (%) above threshold (negative EPS ignored)"),
        ("change_in_percent", "Price change in percent (%) terms"),
        ("pe", "Price/Earnings ratio"),
        ("threshold_dy", "Transition from DY lower than threshold to greater"),
        ("threshold_eps", "Transition from EPS lower than threshold to greater"),
    )
    threshold = forms.FloatField(
        required=True, min_value=0.0, max_value=10000.0, initial=50.0
    )
    timeframe_in_days = forms.IntegerField(
        required=True, min_value=1, max_value=365, initial=7, label="Timeframe (days)"
    )
    show_increasing = forms.BooleanField(
        required=False, initial=True, label="Increasing"
    )
    show_decreasing = forms.BooleanField(
        required=False, initial=True, label="Decreasing"
    )
    max_price = forms.FloatField(required=False, max_value=10000.0)
    metric = forms.ChoiceField(
        required=True, choices=METRIC_CHOICES, initial="change_in_percent"
    )


class SectorSentimentSearchForm(forms.Form):
    normalisation_choices = (
        (1, "None"),
        (2, "Min/Max. scaling"),
        (3, "Divide by maximum"),
    )
    sector = forms.ChoiceField(required=True, choices=CompanySearchForm.SECTOR_CHOICES)
    normalisation_method = forms.ChoiceField(
        required=True, choices=normalisation_choices
    )
    n_days = forms.IntegerField(required=True, max_value=200, min_value=10, initial=30)


class OptimisePortfolioForm(forms.Form):
    method_choices = (
        ("hrp", "Hierarchical Risk Parity"),
        ("ef-sharpe", "Efficient Frontier - Max Sharpe"),
        ("ef-risk", "Efficient Frontier - efficient risk"),
        ("ef-minvol", "Efficient Frontier - minimum volatility"),
    )
    n_days = forms.IntegerField(
        required=True,
        max_value=1000,
        min_value=7,
        initial=180,
        label="Timeframe for data to use (days relative to now)",
    )
    method = forms.ChoiceField(
        required=True,
        choices=method_choices,
        initial=method_choices[0][0],
        label="Optimisation method to use",
    )
    excluded_stocks = forms.MultipleChoiceField(
        choices=(),
        required=False,
        widget=forms.SelectMultiple(attrs={"size": 10}),
        label="Stocks to exclude from optimisation",
    )
    exclude_price = forms.FloatField(
        required=False,
        min_value=0.001,
        max_value=100000.0,
        label="Minimum stock price required at start/end (may be blank)",
    )
    portfolio_cost = forms.IntegerField(
        initial=100 * 1000,
        required=True,
        min_value=1000,
        label="Show share portfolio having dollar value: ($AUD)",
    )
    max_stocks = forms.IntegerField(
        min_value=10,
        max_value=200,
        initial=80,
        label="Maximum stocks to consider (random sample taken)",
    )
    returns_by = forms.ChoiceField(
        choices=(
            ("by_prices", "By ASX Prices for chosen stocks"),
            ("by_black_litterman", "Black-Litterman model"),
        ),
        required=True,
    )
    exclude_etfs = forms.BooleanField(
        required=False, initial=False, label="exclude ETFs?"
    )

    def __init__(self, excluded_stocks, **kwargs):
        super(OptimisePortfolioForm, self).__init__(**kwargs)
        if excluded_stocks is not None:
            self.fields["excluded_stocks"].choices = [(s, s) for s in excluded_stocks]


class OptimiseSectorForm(OptimisePortfolioForm):
    sector = forms.ChoiceField(
        choices=CompanySearchForm.SECTOR_CHOICES,
        required=True,
        initial=CompanySearchForm.SECTOR_CHOICES[0][0],
    )

    def __init__(self, *args, **kwargs):
        sector = kwargs.pop("sector", None)
        super(OptimiseSectorForm, self).__init__(*args, **kwargs)
        if sector:
            self.fields["sector"].initial = sector


class MarketCapSearchForm(forms.Form):
    min_cap = forms.IntegerField(
        min_value=0, initial=10, label="Minimum market cap ($AUD millions)"
    )
    max_cap = forms.IntegerField(
        min_value=0, initial=100, label="Maximum market cap ($AUD millions)"
    )


class FinancialMetricSearchForm(forms.Form):
    """
    Perform search over available balance sheet, earnings, cashflow or other data
    """

    UNIT_CHOICES = (
        ("%", "Percentage (%)"),
        ("Value", "Value"),
        ("Value(M)", "Value (millions)"),
        ("Ratio(max)", "Fraction of maximum value seen"),
    )
    RELATION_CHOICES = (
        ("<=", "Any value (over the years) less than or equal to"),
        (">=", "Any value (over the years) greater than or equal to"),
        ("average <=", "Average (over the years) less than or equal to"),
        ("min <=", "Minimum (over the years) less than or equal to"),
        ("max <=", "Maxmimum (over the years) less than or equal to"),
    )
    metric = forms.ChoiceField(choices=(), required=True, label="Find stocks with... ")
    amount = forms.FloatField(required=True, initial=10.0, label="and value ...")
    relation = forms.ChoiceField(choices=RELATION_CHOICES, label="which matches...")
    unit = forms.ChoiceField(
        choices=UNIT_CHOICES,
        required=True,
        initial="%",
        label="interpreting value as... ",
    )

    def __init__(self, *args, **kwargs):
        metric_choices = [
            (v, v)
            for v in sorted(
                list(
                    CompanyFinancialMetric.objects.order_by("name")
                    .values_list("name", flat=True)
                    .distinct()
                )
            )
        ]
        super().__init__(*args, **kwargs)
        self.fields["metric"].choices = metric_choices


class MomentumSearchForm(forms.Form):
    n_days = forms.IntegerField(
        max_value=300,
        min_value=7,
        required=True,
        initial=300,
        label="Only report cross-overs in past ... days",
    )
    period1 = forms.IntegerField(
        max_value=300,
        min_value=5,
        required=True,
        initial=20,
        label="Number of days to use for MA20",
    )
    period2 = forms.IntegerField(
        max_value=300,
        min_value=20,
        required=True,
        initial=200,
        label="Number of days to use for MA200",
    )
    what_to_search = forms.ChoiceField(
        choices=(("all_stocks", "All stocks"), ("watchlist", "Watchlist")),
        label="Consider .. stocks only",
        widget=forms.RadioSelect,
        required=True,
    )
