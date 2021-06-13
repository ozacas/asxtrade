from django.shortcuts import render
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic.list import MultipleObjectTemplateResponseMixin
from django.views.generic.edit import FormView
from django.shortcuts import render
from abs.forms import ABSDataflowForm
from abs.models import data


def abs_index_view(request):
    raise Http404("Not implemented yet")


class ABSHeadlineView(LoginRequiredMixin, FormView):
    form_class = ABSDataflowForm
    template_name = "abs_dataset.html"

    def form_valid(self, form):
        context = self.get_context_data()
        data_df = data(form.cleaned_data["dataflow"])
        return render(self.request, template_name=self.template_name, context=context)


abs_headlines = ABSHeadlineView.as_view()
