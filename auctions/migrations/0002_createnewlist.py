# Generated by Django 3.0.2 on 2020-09-10 14:01

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('auctions', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='Createnewlist',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=64)),
                ('description', models.CharField(max_length=200)),
                ('start_bid', models.IntegerField()),
                ('bidder', models.CharField(max_length=64)),
                ('category', models.CharField(max_length=64)),
                ('image_url', models.ImageField(blank=True, null=True, upload_to='')),
                ('default', models.BooleanField(blank=True, default=False, null=True)),
            ],
        ),
    ]
