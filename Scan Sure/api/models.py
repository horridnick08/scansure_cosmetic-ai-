from mongoengine import Document, StringField, FloatField, DateTimeField, ListField
from datetime import datetime

class ScanResult(Document):
    # This class inherits from Document, not models.Model 
    # Id is automatically injected by MongoEngine as a MongoDB ObjectId
    image_path = StringField(required=True)
    prediction_label = StringField(required=True)
    confidence_score = FloatField(required=True)
    
    # OCR Extracted Fields
    brand = StringField(null=True, blank=True)
    product = StringField(null=True, blank=True)
    ingredients = ListField(StringField(), default=list)
    barcode = StringField(null=True, blank=True)
    batch = StringField(null=True, blank=True)
    raw_text = StringField(null=True, blank=True)
    
    timestamp = DateTimeField(default=datetime.utcnow)
    
    meta = {
        'collection': 'scan_results'
    }
