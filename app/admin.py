from django.contrib import admin
from .models import *
# Register your models here.

admin.site.register(Chat)
admin.site.register(CsvFile)
admin.site.register(EmployeeDetails)
admin.site.register(Taskallocation)
class PerformanceAdmin(admin.ModelAdmin):
    list_display = ('name', 'work_hours', 'idle_time', 'projects_done', 'mood', 'work_station', 'age', 'experience', 'date')
    search_fields = ('name', 'mood')  # Add any other fields you want to search
    fields = ('name', 'work_hours', 'idle_time', 'projects_done', 'mood', 'work_station', 'age', 'experience', 'date')

admin.site.register(Performance, PerformanceAdmin)
admin.site.register(Task)