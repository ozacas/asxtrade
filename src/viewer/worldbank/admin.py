from django.contrib import admin
from worldbank.models import (
    WorldBankCountry,
    WorldBankTopic,
    WorldBankIndicators,
    WorldBankInvertedIndex,
    WorldBankDataCache
)

@admin.register(WorldBankDataCache)
class WorldBankDataCacheAdmin(admin.ModelAdmin):
    list_display= ('tag', 'last_updated')

@admin.register(WorldBankCountry)
class WorldBankCountriesAdmin(admin.ModelAdmin):
    list_display = ('country_code', 'name', 'last_updated')

@admin.register(WorldBankTopic)
class WorldBankTopicsAdmin(admin.ModelAdmin):
    list_display = ('id', 'topic', 'last_updated', 'source_note')

@admin.register(WorldBankIndicators)
class WorldBankIndicatorsAdmin(admin.ModelAdmin):
    list_display = ('id', 'wb_id', 'last_updated', 'name', 'source_organisation')
    exclude = ('topics',)

@admin.register(WorldBankInvertedIndex)
class WorldBankInvertedIndexAdmin(admin.ModelAdmin):
    list_display = ('id', 'country', 'topic_id', 'topic_name', 'n_attributes',
                    'last_updated', 'tag', 'indicator_id', 'indicator_name')
