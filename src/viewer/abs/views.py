from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.list import MultipleObjectTemplateResponseMixin
from django.views.generic.edit import FormView
from django.http import Http404
from django.shortcuts import render
from abs.forms import ABSDataflowForm
from app.data import cache_plot
from abs.models import data
from app.plots import user_theme
import pandas as pd
import plotnine as p9


def abs_index_view(request):
    raise Http404("Not implemented yet")


class ABSHeadlineView(LoginRequiredMixin, FormView):
    form_class = ABSDataflowForm
    template_name = "abs_dataset.html"

    def plot_abs_dataframe(self, df: pd.DataFrame) -> p9.ggplot:
        title = ""
        facets = []
        n_per_facet = {}
        for col in df.columns:
            try:
                n_values = df[col].nunique()
            except:
                print(f"Ignoring unusable column: {col}")
                continue
            if n_values == 1 and col not in [
                "TIME_PERIOD",
                "value",
                "Measure",
                "OBS_COMMENT",
            ]:
                title += f"{col}={df.at[0, col]}"
            elif n_values > 1 and col not in ["value", "TIME_PERIOD", "OBS_COMMENT"]:
                facets.append(col)
                n_per_facet[col] = n_values
        extra_args = {}

        if len(facets) > 2:
            # can only use two variables as plotting facets, third value will be used as a group on each plot
            # any more facets is not supported at this stage
            sorted_facets = sorted(n_per_facet.keys(), key=lambda k: n_per_facet[k])
            print(n_per_facet)
            print(sorted_facets)
            facets = sorted_facets[-2:]
            extra_args.update({"group": sorted_facets[0], "color": facets[0]})
            print(f"Using {facets} as facets, {extra_args} as series")

        plot = p9.ggplot(
            df, p9.aes(x="TIME_PERIOD", y="value", **extra_args)
        ) + p9.geom_point(size=3)

        if len(facets) > 0 and len(facets) <= 2:
            plot += p9.facet_wrap(
                "~" + " + ".join(facets[:2]), ncol=len(facets), scales="free_y"
            )

        # compute figure size to give enough room for each plot
        mult = 1
        for facet in facets:
            mult *= n_per_facet[facet]
        mult /= len(facets)
        nrow = int(mult + 1)

        plot_theme = {
            "figure_size": (12, int(nrow * 1.5)),
            "asxtrade_want_fill_d": True,
        }
        if (
            len(facets) == 2
        ):  # two columns of plots? if so, make sure  space for axis labels
            plot_theme.update({"subplots_adjust": {"wspace": 0.2}})
        return user_theme(plot, title=title, **plot_theme)

    def form_valid(self, form):
        context = self.get_context_data()
        key = form.cleaned_data["dataflow"]
        data_df = data(key)
        context.update(
            {
                "plot_title": key,
                "plot_uri": cache_plot(
                    f"{key}-abs-plot",
                    lambda: self.plot_abs_dataframe(data_df),
                    dont_cache=True,
                ),
            }
        )
        return render(self.request, template_name=self.template_name, context=context)


abs_headlines = ABSHeadlineView.as_view()
