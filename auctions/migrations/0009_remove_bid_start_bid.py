# Generated by Django 3.0.2 on 2020-09-12 07:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0008_remove_bid_owner'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='bid',
            name='start_bid',
        ),
    ]
