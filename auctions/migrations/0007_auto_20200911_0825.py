# Generated by Django 3.0.2 on 2020-09-11 02:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0006_auto_20200911_0625'),
    ]

    operations = [
        migrations.AlterField(
            model_name='createnewlist',
            name='image',
            field=models.ImageField(blank=True, default='', null=True, upload_to='items'),
        ),
    ]
