from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings
from .models import ScanResult
import os
import uuid
from ocr import process_image

class ScanAPIView(APIView):
    def post(self, request):
        try:
            image_file = request.FILES.get('image')
            prediction_label = request.data.get('prediction_label')
            confidence_score = request.data.get('confidence_score')

            # Validation
            if not all([image_file, prediction_label, confidence_score]):
                return Response(
                    {"error": "Missing required fields (image, prediction_label, confidence_score)."},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Manual file handling setup
            file_extension = os.path.splitext(image_file.name)[1]
            unique_name = f"{uuid.uuid4()}{file_extension}"
            
            media_path = os.path.join(settings.MEDIA_ROOT, 'scans')
            os.makedirs(media_path, exist_ok=True) # create local folder if omitted
            
            save_path = os.path.join(media_path, unique_name)
            
            # Save file inside media/scans/ via raw file handling
            with open(save_path, 'wb+') as destination:
                for chunk in image_file.chunks():
                    destination.write(chunk)

            # Image path relative to media directory
            image_relative_path = f"scans/{unique_name}"

            # Process OCR extraction
            extracted_data = process_image(save_path)

            # Save Document in MongoDB via MongoEngine directly without ORM models.save()
            scan_record = ScanResult(
                image_path=image_relative_path,
                prediction_label=str(prediction_label),
                confidence_score=float(confidence_score),
                brand=extracted_data.get("brand"),
                product=extracted_data.get("product"),
                ingredients=extracted_data.get("ingredients"),
                barcode=extracted_data.get("barcode"),
                batch=extracted_data.get("batch"),
                raw_text=extracted_data.get("raw_text")
            )
            scan_record.save()

            return Response({
                "message": "Scan record saved successfully.",
                "id": str(scan_record.id),
                "image_path": image_relative_path,
                "prediction_label": scan_record.prediction_label,
                "confidence_score": scan_record.confidence_score,
                "brand": scan_record.brand,
                "product": scan_record.product,
                "ingredients": scan_record.ingredients,
                "barcode": scan_record.barcode,
                "batch": scan_record.batch,
                # omitting raw_text from response as it could be huge
                "timestamp": scan_record.timestamp
            }, status=status.HTTP_201_CREATED)

        except ValueError:
            return Response({"error": "Invalid format. confidence_score must be a float."}, status=status.HTTP_400_BAD_REQUEST)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def get(self, request):
        try:
            # Query all elements and order by timestamp descending
            scans = ScanResult.objects.all().order_by('-timestamp')
            
            data = []
            for scan in scans:
                data.append({
                    "id": str(scan.id),
                    "image_path": scan.image_path,
                    "prediction_label": scan.prediction_label,
                    "confidence_score": scan.confidence_score,
                    "brand": getattr(scan, 'brand', None),
                    "product": getattr(scan, 'product', None),
                    "ingredients": getattr(scan, 'ingredients', []),
                    "barcode": getattr(scan, 'barcode', None),
                    "batch": getattr(scan, 'batch', None),
                    "timestamp": scan.timestamp.isoformat() if scan.timestamp else None
                })

            return Response(data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
