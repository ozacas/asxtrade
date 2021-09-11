from django.contrib import admin
from ecb.models import ECBDataCache, ECBFlow, ECBMetadata


@admin.register(ECBDataCache)
class ECBDataCacheAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "tag",
        "last_updated",
        "size_in_bytes",
        "status",
        "last_updated",
        "sha256",
        "scope",
    )
    exclude = ("dataframe",)


@admin.register(ECBMetadata)
class ECBMetadataAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "metadata_type",
        "flow",
        "code",
        "column_name",
        "printable_code",
    )


@admin.register(ECBFlow)
class ECBFlowAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "name",
        "description",
        "is_test_data",
        "last_updated",
        "prepared",
        "sender",
        "source",
    )
