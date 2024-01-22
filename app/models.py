from django.db import models

class Chat(models.Model):
    user_input = models.TextField()
    response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)
    
    
    
class CsvFile(models.Model):
    file = models.FileField(upload_to='uploads/')
    
    
class EmployeeDetails(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    experience = models.PositiveIntegerField(null=True, blank=True)
    skills = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.name    

class Performance(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    work_hours = models.FloatField(null=True, blank=True)
    idle_time = models.FloatField(null=True, blank=True)
    projects_done = models.IntegerField(null=True, blank=True)
    mood = models.CharField(max_length=255, null=True, blank=True)
    work_station = models.IntegerField(null=True, blank=True)
    age = models.PositiveIntegerField(null=True, blank=True)
    experience = models.PositiveIntegerField(null=True, blank=True)
    date = models.DateTimeField(null=True, blank=True)
    
    def __str__(self):
        return self.name
    
class Taskallocation(models.Model):
    name = models.CharField(max_length=255, null=True, blank=True)
    skills = models.TextField(null=True, blank=True)
    task_given = models.CharField(max_length=255, null=True, blank=True)
    completed_in = models.FloatField(null=True, blank=True)

from django.db import models

class Task(models.Model):
    task_id = models.AutoField(primary_key=True)
    deadline = models.DateField()
    dependencies = models.JSONField(default=list)
    resource_availability = models.IntegerField()
    team_performance = models.FloatField()
    actual_completion_time = models.IntegerField()