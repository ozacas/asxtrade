from app.models import Quotation, VirtualPurchase
from bson.objectid import ObjectId
import traceback

class VirtualPurchaseMixin:
    """
    Retrieve the object by mongo _id for use by CRUD CBV views for VirtualPurchase's
    """

    def get_object(self, queryset=None):
        slug = self.kwargs.get("slug")
        purchase = VirtualPurchase.objects.mongo_find_one({"_id": ObjectId(slug)})
        # print(purchase)
        purchase["id"] = purchase["_id"]
        purchase.pop("_id", None)
        return VirtualPurchase(**purchase)


class SearchMixin:
    model = Quotation
    object_list = Quotation.objects.none()

    def get(self, request, *args, **kwargs):
        """need to subclass this method to ensure pagination works correctly (as 'next', 'last' etc. is GET not POST)"""
        d = {}
        key = self.__class__.__name__
        # print("Updating session state: {}".format(key))
        d.update(request.session.get(key, {}))  # update the form to the session state
        return self.update_form(d)

    def get_initial_form(self, form_values):
        assert isinstance(form_values, dict)
        form_class = self.get_form_class()
        return form_class(initial=form_values, label_suffix="")

    def update_form(self, form_values):
        assert isinstance(form_values, dict)
        # apply the form settings to self.queryset (specific to a CBV - watch for subclass overrides)
        try:
            self.object_list = self.get_queryset(**form_values)
        except Exception:
            traceback.format_exc()
            self.object_list = [] # defend against broken get_queryset() implementations

        state_field = (
            self.__class__.__name__
        )  # NB: must use class name so that each search type has its own state for a given user
        self.request.session[state_field] = form_values
        context = self.get_context_data()
        assert context is not None
        assert self.action_url is not None
        context["action_url"] = self.action_url
        self.form = self.form_class(initial=form_values, label_suffix="")
        context["form"] = self.form
        return self.render_to_response(context)

    def form_invalid(self, form):
        print(f"Invalid form data: {form.cleaned_data}")
        return self.update_form(form.cleaned_data)

    # this is called from self.post()
    def form_valid(self, form):
        assert form.is_valid()
        return self.update_form(form.cleaned_data)
