from django.db import models

# Create models.

class User(models.Model):
    age = models.IntegerField()
    name = models.CharField(max_length=30)

    def __str__(self):
        return self.name
