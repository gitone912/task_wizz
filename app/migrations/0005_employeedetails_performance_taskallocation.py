# Generated by Django 4.1.11 on 2023-12-27 23:15

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0004_csvfile_delete_calmsongs_delete_happysongs_and_more"),
    ]

    operations = [
        migrations.CreateModel(
            name="EmployeeDetails",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(blank=True, max_length=255, null=True)),
                ("age", models.PositiveIntegerField(blank=True, null=True)),
                ("experience", models.PositiveIntegerField(blank=True, null=True)),
                ("skills", models.TextField(blank=True, null=True)),
            ],
        ),
        migrations.CreateModel(
            name="Performance",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(blank=True, max_length=255, null=True)),
                ("work_hours", models.FloatField(blank=True, null=True)),
                ("idle_time", models.FloatField(blank=True, null=True)),
                ("projects_done", models.IntegerField(blank=True, null=True)),
                ("mood", models.CharField(blank=True, max_length=255, null=True)),
                ("work_station", models.IntegerField(blank=True, null=True)),
                ("age", models.PositiveIntegerField(blank=True, null=True)),
                ("experience", models.PositiveIntegerField(blank=True, null=True)),
                ("date", models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name="Taskallocation",
            fields=[
                (
                    "id",
                    models.BigAutoField(
                        auto_created=True,
                        primary_key=True,
                        serialize=False,
                        verbose_name="ID",
                    ),
                ),
                ("name", models.CharField(blank=True, max_length=255, null=True)),
                ("skills", models.TextField(blank=True, null=True)),
            ],
        ),
    ]
