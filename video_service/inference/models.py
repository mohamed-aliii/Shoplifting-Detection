from django.db import models

class VideoUpload(models.Model):
    video = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    processed = models.BooleanField(default=False)
    is_shoplifting = models.BooleanField(null=True)
    confidence = models.FloatField(null=True)

    def __str__(self):
        return f"Video {self.id} - {'Processed' if self.processed else 'Pending'}"
