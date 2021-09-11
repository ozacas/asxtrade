"""
Responsible for views related to VirtualPurchase (ie. fake stock purchases at a particular date) models
"""
from django import forms
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import CreateView, UpdateView, DeleteView
from app.mixins import VirtualPurchaseMixin
from app.models import VirtualPurchase, validate_user, validate_stock, latest_quote
from app.messages import info, warning

class BuyVirtualStock(LoginRequiredMixin, CreateView):
    model = VirtualPurchase
    success_url = "/show/watched"
    form_class = forms.models.modelform_factory(
        VirtualPurchase,
        fields=["asx_code", "buy_date", "price_at_buy_date", "amount", "n"],
        widgets={"asx_code": forms.TextInput(attrs={"readonly": "readonly"})},
    )

    def form_valid(self, form):
        req = self.request
        resp = super().form_valid(form)  # only if no exception raised do we save...
        self.object = form.save(commit=False)
        self.object.user = validate_user(req.user)
        self.object.save()
        info(req, "Saved purchase of {}".format(self.kwargs.get("stock")))
        return resp

    def get_context_data(self, **kwargs):
        result = super().get_context_data(**kwargs)
        result["title"] = "Add {} purchase to watchlist".format(
            self.kwargs.get("stock")
        )
        return result

    def get_initial(self, **kwargs):
        stock = kwargs.get("stock", self.kwargs.get("stock"))
        amount = kwargs.get("amount", self.kwargs.get("amount", 5000.0))
        user = self.request.user
        validate_stock(stock)
        validate_user(user)
        quote, latest_date = latest_quote(stock)
        cur_price = quote.last_price
        if cur_price >= 1e-6:
            return {
                "asx_code": stock,
                "user": user,
                "buy_date": latest_date,
                "price_at_buy_date": cur_price,
                "amount": amount,
                "n": int(amount / cur_price),
            }
        else:
            warning(
                self.request, "Cannot buy {} as its price is zero/unknown".format(stock)
            )
            return {}


buy_virtual_stock = BuyVirtualStock.as_view()



class EditVirtualStock(LoginRequiredMixin, VirtualPurchaseMixin, UpdateView):
    model = VirtualPurchase
    success_url = "/show/watched"
    form_class = forms.models.modelform_factory(
        VirtualPurchase,
        fields=["asx_code", "buy_date", "price_at_buy_date", "amount", "n"],
        widgets={
            "asx_code": forms.TextInput(attrs={"readonly": "readonly"}),
            "buy_date": forms.DateInput(),
        },
    )


edit_virtual_stock = EditVirtualStock.as_view()


class DeleteVirtualPurchaseView(LoginRequiredMixin, VirtualPurchaseMixin, DeleteView):
    model = VirtualPurchase
    success_url = "/show/watched"

delete_virtual_stock = DeleteVirtualPurchaseView.as_view()
