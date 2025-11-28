"""
Video Translation API

영상 자막 STT 및 번역 엔드포인트
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session
from uuid import UUID
from pathlib import Path

# Database
from app.database import get_db

# Services
from app.services.video_translation_service import VideoTranslationService

# Schemas
from app.schemas.video_translation import (
    VideoSTTRequest,
    VideoSTTResponse,
    VideoTranslationRequest,
    VideoTranslationResponse,
    SubtitleDownloadResponse,
    MultilingualSubtitlesResponse
)

# Auth (추후 구현)
# from app.api.dependencies import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ai/video", tags=["video-translation"])


@router.post("/stt", response_model=VideoSTTResponse, status_code=status.HTTP_201_CREATED)
async def process_video_stt(
    request: VideoSTTRequest,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # 추후 인증 추가
):
    """
    영상 STT 처리 (Whisper API)

    영상 파일에서 음성을 인식하여 타임스탬프가 포함된 자막을 생성합니다.

    **처리 과정**:
    1. ffmpeg로 영상에서 오디오 추출
    2. OpenAI Whisper API로 STT 수행
    3. 타임스탬프 세그먼트 DB 저장

    **요청 예시**:
    ```json
    {
      "video_file_id": "123e4567-e89b-12d3-a456-426614174000",
      "source_language": "ko"
    }
    ```

    **응답 예시**:
    ```json
    {
      "subtitle_id": "...",
      "video_file_id": "...",
      "language": "ko",
      "subtitle_type": "ORIGINAL",
      "segments": [
        {
          "sequence_number": 1,
          "start_time_ms": 0,
          "end_time_ms": 3500,
          "text": "안녕하세요, 오늘은 인공지능에 대해 말씀드리겠습니다.",
          "confidence": 0.95
        }
      ],
      "total_segments": 10,
      "created_at": "2025-01-17T10:30:00Z"
    }
    ```
    """
    logger.info(f"📥 STT 요청: video_file={request.video_file_id}, lang={request.source_language}")

    try:
        service = VideoTranslationService()

        # 임시 사용자 ID (추후 인증 연동)
        user_id = UUID("00000000-0000-0000-0000-000000000000")

        result = await service.process_stt(
            video_file_id=request.video_file_id,
            source_language=request.source_language,
            user_id=user_id,
            db=db
        )

        return VideoSTTResponse(**result)

    except FileNotFoundError as e:
        logger.error(f"❌ 파일 없음: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except ValueError as e:
        logger.error(f"❌ 유효성 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ STT 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"STT 처리 중 오류 발생: {str(e)}"
        )


@router.post("/translate", response_model=VideoTranslationResponse, status_code=status.HTTP_201_CREATED)
async def translate_video_subtitle(
    request: VideoTranslationRequest,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # 추후 인증 추가
):
    """
    영상 자막 번역

    STT로 생성된 원본 자막을 목표 언어로 번역합니다.
    프로젝트 문서를 컨텍스트로 활용하여 더 정확한 번역을 제공합니다.

    **처리 과정**:
    1. 원본 자막 조회 (STT 결과)
    2. 컨텍스트 문서 조회 (선택사항)
    3. 각 세그먼트 번역 (ContextEnhancedTranslationAgent)
    4. 번역된 자막 DB 저장

    **요청 예시**:
    ```json
    {
      "video_file_id": "123e4567-e89b-12d3-a456-426614174000",
      "document_ids": ["789...", "012..."],
      "source_language": "ko",
      "target_language": "en"
    }
    ```

    **응답 예시**:
    ```json
    {
      "subtitle_id": "...",
      "video_file_id": "...",
      "source_language": "ko",
      "target_language": "en",
      "subtitle_type": "TRANSLATED",
      "segments": [
        {
          "sequence_number": 1,
          "start_time_ms": 0,
          "end_time_ms": 3500,
          "text": "Hello, today I will talk about artificial intelligence.",
          "confidence": 0.95
        }
      ],
      "total_segments": 10,
      "context_used": true,
      "context_document_count": 2,
      "created_at": "2025-01-17T10:35:00Z"
    }
    ```
    """
    logger.info(
        f"📥 자막 번역 요청: video={request.video_file_id}, "
        f"{request.source_language} → {request.target_language}"
    )

    try:
        service = VideoTranslationService()

        # 임시 사용자 ID (추후 인증 연동)
        user_id = UUID("00000000-0000-0000-0000-000000000000")

        result = await service.process_translation(
            video_file_id=request.video_file_id,
            document_ids=request.document_ids,
            source_language=request.source_language,
            target_language=request.target_language,
            user_id=user_id,
            db=db
        )

        return VideoTranslationResponse(**result)

    except ValueError as e:
        logger.error(f"❌ 유효성 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ 번역 처리 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"번역 처리 중 오류 발생: {str(e)}"
        )


@router.get("/subtitles/{video_file_id}", response_model=MultilingualSubtitlesResponse)
async def get_video_subtitles(
    video_file_id: UUID,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # 추후 인증 추가
):
    """
    다국어 자막 조회

    영상의 모든 언어 자막을 조회합니다.
    원본 언어와 번역된 모든 언어의 자막이 포함됩니다.

    **응답 예시**:
    ```json
    {
      "video_file_id": "123e4567-e89b-12d3-a456-426614174000",
      "original_language": "ko",
      "available_languages": ["ko", "en", "ja"],
      "segments": [
        {
          "sequence_number": 1,
          "start_time_ms": 0,
          "end_time_ms": 3500,
          "original_text": "안녕하세요",
          "translations": {
            "en": "Hello",
            "ja": "こんにちは"
          },
          "confidence": 0.95,
          "detected_terms": []
        }
      ],
      "total_segments": 10
    }
    ```
    """
    logger.info(f"📥 다국어 자막 조회 요청: video_file_id={video_file_id}")

    try:
        service = VideoTranslationService()

        result = await service.get_subtitles(
            video_file_id=video_file_id,
            db=db
        )

        return MultilingualSubtitlesResponse(**result)

    except ValueError as e:
        logger.error(f"❌ 유효성 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ 자막 조회 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"자막 조회 중 오류 발생: {str(e)}"
        )


@router.get("/subtitle/{subtitle_id}/download", response_class=FileResponse)
async def download_subtitle_file(
    subtitle_id: UUID,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # 추후 인증 추가
):
    """
    자막 파일 다운로드 (SRT 형식)

    저장된 자막을 SRT 형식 파일로 다운로드합니다.
    파일이 없으면 자동 생성됩니다.

    **응답**:
    - Content-Type: `application/x-subrip`
    - Content-Disposition: `attachment; filename="subtitle.srt"`
    """
    logger.info(f"📥 자막 다운로드 요청: subtitle_id={subtitle_id}")

    try:
        service = VideoTranslationService()

        # SRT 파일 생성 (또는 기존 파일 반환)
        result = await service.generate_subtitle_file(
            subtitle_id=subtitle_id,
            db=db
        )

        file_path = Path(result["file_path"])

        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="자막 파일을 찾을 수 없습니다"
            )

        # 파일명 생성
        filename = f"subtitle_{result['language']}_{result['subtitle_type']}.srt"

        logger.info(f"✅ 자막 다운로드: {file_path}")

        return FileResponse(
            path=str(file_path),
            media_type="application/x-subrip",
            filename=filename
        )

    except ValueError as e:
        logger.error(f"❌ 유효성 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ 파일 다운로드 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"파일 다운로드 중 오류 발생: {str(e)}"
        )


@router.get("/subtitle/{subtitle_id}/info", response_model=SubtitleDownloadResponse)
async def get_subtitle_info(
    subtitle_id: UUID,
    db: Session = Depends(get_db)
    # current_user = Depends(get_current_user)  # 추후 인증 추가
):
    """
    자막 정보 조회

    자막 파일 메타데이터를 조회합니다.
    """
    logger.info(f"📥 자막 정보 조회: subtitle_id={subtitle_id}")

    try:
        service = VideoTranslationService()

        result = await service.generate_subtitle_file(
            subtitle_id=subtitle_id,
            db=db
        )

        return SubtitleDownloadResponse(**result)

    except ValueError as e:
        logger.error(f"❌ 유효성 검증 실패: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"❌ 정보 조회 실패: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"정보 조회 중 오류 발생: {str(e)}"
        )
