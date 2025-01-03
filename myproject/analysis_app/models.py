from django.db import models

# Create your models here.
class DiaryEntry(models.Model):
    date = models.DateField()
    text = models.TextField()

    def __str__(self):
        return f"Diary Entry on {self.date}"