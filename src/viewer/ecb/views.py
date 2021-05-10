import re
from django.http import Http404
from django.shortcuts import render
from app.models import validate_user
from app.data import cache_plot
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
            [flow for flow in ECBFlow.objects.filter(is_test_data=False)],
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
        print("in form_valid")
        print(form.cleaned_data)

        df = fetch_dataframe(self.data_flow.name)
        if df is None or len(df) == 0:
            raise Http404(f"Unable to load dataframe: {self.data_flow}")

        for k, v in form.cleaned_data.items():
            k = k[len("dimension_") :]
            df = df[df[k] == v]
            if len(df) == 0:
                warning(self.request, f"No rows of data left after filtering: {k} {v}")
                break

        plot = None
        if len(df) > 0:
            title, x_axis_column, y_axis_column, df = detect_dataframe(
                df, self.data_flow
            )
            plot = (
                p9.ggplot(df, p9.aes(x=x_axis_column, y=y_axis_column))
                + p9.geom_point()
                + p9.geom_line()
                + p9.labs(title=title, x="", y="")
                + p9.theme(figure_size=(12, 6))
            )

        context = self.get_context_data()
        cache_key = "-".join(sorted(form.cleaned_data.values()))
        context.update(
            {
                "dataflow": self.data_flow,
                "dataflow_name": self.data_flow.name,
                "plot_uri": cache_plot(cache_key, lambda: plot),
            }
        )
        return render(self.request, self.template_name, context)