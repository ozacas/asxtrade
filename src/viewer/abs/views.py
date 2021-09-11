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
    fixed_datapoints = None  # set used to perform de-dupe

    def plot_abs_dataframe(self, df: pd.DataFrame) -> p9.ggplot:
        facets = []
        n_per_facet = {}
        print(df)
        for col in df.columns:
            try:
                n_values = df[col].nunique()
                if n_values == 1 and col not in [
                    "TIME_PERIOD",
                    "value",
                    "Measure",
                    "OBS_COMMENT",
                ]:
                    self.fixed_datapoints.add(f"{col}={df.at[0, col]}")
                elif n_values > 1 and col not in [
                    "value",
                    "TIME_PERIOD",
                    "OBS_COMMENT",
                ]:
                    facets.append(col)
                    n_per_facet[col] = n_values
            except:
                print(f"Ignoring unusable column: {col}")
                continue

        extra_args = {}
        need_shape = False
        if len(facets) > 2:
            # can only use two variables as plotting facets, third value will be used as a group on each plot
            # any more facets is not supported at this stage
            sorted_facets = sorted(n_per_facet.keys(), key=lambda k: n_per_facet[k])
            # print(n_per_facet)
            # print(sorted_facets)
            facets = sorted_facets[-2:]
            extra_args.update(
                {
                    "group": sorted_facets[0],
                    "color": facets[0],
                    "shape": sorted_facets[0],
                }
            )
            need_shape = True
            print(f"Using {facets} as facets, {extra_args} as series")
        else:
            if len(facets) > 0:
                extra_args.update({"color": facets[0]})

        # compute figure size to give enough room for each plot
        mult = 1
        for facet in facets:
            mult *= n_per_facet[facet]
        mult /= len(facets)
        nrow = int(mult + 1)

        # facet column names must not have spaces in them as this is not permitted by plotnine facet formulas
        if len(facets) > 0:
            new_facets = []
            for f in facets:
                if " " in f:
                    new_name = f.replace(" ", "_")
                    df = df.rename(columns={f: new_name})
                    new_facets.append(new_name)
                else:
                    new_facets.append(f)
            facets = new_facets
            if "color" in extra_args:
                extra_args.update({"color": facets[0]})
            print(f"Renamed facet columns due to whitespace: {facets}")

        plot = p9.ggplot(
            df, p9.aes(x="TIME_PERIOD", y="value", **extra_args)
        ) + p9.geom_point(size=3)

        if len(facets) > 0 and len(facets) <= 2:
            facet_str = "~" + " + ".join(facets[:2])
            print(f"Using facet formula: {facet_str}")
            plot += p9.facet_wrap(facet_str, ncol=len(facets), scales="free_y")

        plot_theme = {
            "figure_size": (12, int(nrow * 1.5)),
        }
        if (
            len(facets) == 2
        ):  # two columns of plots? if so, make sure  space for axis labels
            plot_theme.update({"subplots_adjust": {"wspace": 0.2}})
        if need_shape:
            plot += p9.scale_shape(guide="legend")
            plot += p9.guides(
                colour=False
            )  # colour legend is not useful since it is included in the facet title
            plot_theme.update({"legend_position": "right"})
        return user_theme(plot, **plot_theme)

    def form_valid(self, form):
        context = self.get_context_data()
        key = form.cleaned_data["dataflow"]
        data_df = data(key)
        self.fixed_datapoints = set()  # required to perform de-dupe
        context.update(
            {
                "plot_title": key,
                "plot_fixed_datapoints": self.fixed_datapoints,
                "plot_uri": cache_plot(
                    f"{key}-abs-plot",
                    lambda ld: self.plot_abs_dataframe(data_df),
                ),
            }
        )
        return render(self.request, template_name=self.template_name, context=context)


abs_headlines = ABSHeadlineView.as_view()
