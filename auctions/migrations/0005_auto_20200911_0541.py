# Generated by Django 3.0.2 on 2020-09-10 23:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0004_auto_20200911_0446'),
    ]

    operations = [
        migrations.AlterField(
            model_name='bid',
            name='bid_id',
            field=models.IntegerField(),
        ),
        migrations.AlterField(
            model_name='bid',
            name='bidder',
            field=models.CharField(blank=True, default=None, max_length=64, null=True),
        ),
    ]
