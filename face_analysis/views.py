# face_analysis/views.py

from rest_framework.decorators import api_view, parser_classes
from rest_framework.parsers import MultiPartParser
from rest_framework.response import Response
from rest_framework import status
from django.conf import settings

from .utils import is_face_present, is_blurry, is_frontal_face
from .level_model import predict_acne_level

import numpy as np
import cv2
import os
from datetime import datetime

@api_view(['POST'])
@parser_classes([MultiPartParser])
def analyze_faces_and_acne_level(request):
    images = request.FILES.getlist('image')
    if not images or len(images) != 5:
        return Response({'detail': '이미지 5장이 필요합니다.'}, status=status.HTTP_400_BAD_REQUEST)

    upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploaded')
    os.makedirs(upload_dir, exist_ok=True)

    prefix = datetime.now().strftime('%Y%m%d_%H%M%S')
    results = []
    selected_image = None  # 분석에 쓸 유효 이미지

    # 1. 얼굴, 흐림, 정면 체크
    for idx, img_file in enumerate(images):
        img_array = np.frombuffer(img_file.read(), np.uint8)
        image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        filename = f"{prefix}_{idx}.jpg"
        filepath = os.path.join(upload_dir, filename)
        cv2.imwrite(filepath, image)

        face_ok = is_face_present(image)
        blur_ok = not is_blurry(image)
        frontal_ok = is_frontal_face(image) if face_ok else False

        blur_status = "선명" if blur_ok else "흐림"
        face_status = "얼굴" if face_ok else "얼굴없음"
        frontal_status = "정면" if frontal_ok else "비정면"
        print(f"[이미지 저장] 파일명: {filename}, 판별: {blur_status}, {face_status}, {frontal_status}")

        result = {
            "filename": filename,
            "face_detected": face_ok,
            "not_blurry": blur_ok,
            "frontal": frontal_ok
        }
        results.append(result)

        # 조건 만족하는 첫 번째 이미지를 모델 분석에 사용
        if selected_image is None and face_ok and blur_ok and frontal_ok:
            selected_image = image
            selected_filename = filename

    if selected_image is None:
        return Response({
            "detail": "5장 모두에서 적합한 얼굴 사진을 찾지 못했습니다.",
            "results": results
        }, status=422)

    # 2. 피부 상태(여드름 레벨) 모델 분석
    acne_level, prob = predict_acne_level(selected_image)

    # 3. 결과 종합
    level_msg = "정상 (여드름 없음)" if acne_level == 0 else f"여드름 레벨 {acne_level} (1~5)"
    return Response({
        "detail": f"적합한 얼굴 사진 {selected_filename}에 대해 피부 레벨을 판별했습니다.",
        "selected_file": selected_filename,
        "acne_level": int(acne_level),
        "confidence": float(prob),
        "level_msg": level_msg,
        "results": results
    }, status=200)
