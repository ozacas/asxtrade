from django.core.paginator import Paginator

class NoCountPaginator(Paginator):
    @property
    def count(self):
        return 999999999 # Some arbitrarily large number,
                         # so we can still get our page tab.
