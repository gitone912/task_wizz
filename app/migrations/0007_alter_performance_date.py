# Generated by Django 4.1.11 on 2023-12-28 04:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ("app", "0006_taskallocation_completed_in_and_more"),
    ]

    operations = [
        migrations.AlterField(
            model_name="performance",
            name="date",
            field=models.DateTimeField(auto_now_add=True, null=True),
        ),
    ]
