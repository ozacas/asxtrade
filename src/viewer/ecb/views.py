import re
from django.http import Http404
from django.shortcuts import render
from app.models import validate_user
from app.data import cache_plot
from app.plots import user_theme
from app.messages import warning
from django.contrib.auth.decorators import login_required
from ecb.models import ECBFlow, fetch_dataframe, detect_dataframe
from ecb.forms import ECBDynamicDimensionForm
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.edit import FormView
import plotnine as p9


@login_required
def ecb_index_view(request):
    validate_user(request.user)
    context = {
        "dataflows": sorted(
            [
                flow
                for flow in ECBFlow.objects.filter(is_test_data=False).filter(
                    data_available=True
                )
            ],
            key=lambda f: f.description,
        )
    }
    return render(request, "index.html", context)


class ECBDataflowView(LoginRequiredMixin, FormView):
    template_name = "dataflow_view.html"
    form_class = ECBDynamicDimensionForm
    data_flow = None

    def get(self, request, *args, **kwargs):
        df_str = kwargs.get("dataflow", None)
        assert re.match(r"^\w+$", df_str)
        self.data_flow = ECBFlow.objects.filter(name=df_str).first()
        assert self.data_flow is not None

        return super().get(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        df_str = kwargs.pop("dataflow", None)
        assert re.match(r"^\w+$", df_str)
        self.data_flow = ECBFlow.objects.filter(name=df_str).first()
        assert self.data_flow is not None
        return super().post(request, self.data_flow, **kwargs)

    def get_form(self, form_class=None):
        if form_class is None:
            form_class = self.get_form_class()

        kwargs = self.get_form_kwargs()
        return form_class(self.data_flow, **kwargs)

    def form_valid(self, form):
        df = fetch_dataframe(self.data_flow.name)
        if df is None or len(df) == 0:
            raise Http404(f"Unable to load dataframe: {self.data_flow}")

        filter_performance = []
        for k, v in form.cleaned_data.items():
            rows_at_start = len(df)
            print(f"Filtering rows for {k}: total {rows_at_start} rows at start")
            k = k[len("dimension_") :]
            if rows_at_start < 10000:
                unique_values_left = df[k].unique()
            else:
                unique_values_left = set()
            df = df[df[k] == v]
            rows_at_end = len(df)

            filter_performance.append(
                (k, v, rows_at_start, rows_at_end, unique_values_left)
            )
            print(f"After filtering: now {rows_at_end} rows")
            if len(df) == 0:
                warning(self.request, f"No rows of data left after filtering: {k} {v}")
                break

        plot = None
        plot_title = ""
        if len(df) > 0:
            plot_title, x_axis_column, y_axis_column, df = detect_dataframe(
                df, self.data_flow
            )
            plot = (
                p9.ggplot(df, p9.aes(x=x_axis_column, y=y_axis_column))
                + p9.geom_point()
                + p9.geom_line()
            )
            plot = user_theme(plot)

        context = self.get_context_data()
        cache_key = "-".join(sorted(form.cleaned_data.values())) + "-ecb-plot"
        context.update(
            {
                "dataflow": self.data_flow,
                "dataflow_name": self.data_flow.name,
                "filter_performance": filter_performance,
                "plot_title": plot_title,
                "plot_uri": cache_plot(cache_key, lambda: plot),
            }
        )
        return render(self.request, self.template_name, context)