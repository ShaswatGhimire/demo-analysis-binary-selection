# Generated by Django 3.0.2 on 2020-09-12 07:07

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0007_auto_20200911_0825'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bid',
            name='owner',
        ),
    ]
