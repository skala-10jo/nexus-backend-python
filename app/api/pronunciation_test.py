"""
발음 평가 테스트 엔드포인트
Azure Speech Pronunciation Assessment API 테스트용
"""

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import azure.cognitiveservices.speech as speechsdk
import tempfile
import os
import json

from app.config import settings

router = APIRouter(prefix="/api/ai/pronunciation-test", tags=["pronunciation-test"])


class PronunciationResult(BaseModel):
    """발음 평가 결과"""
    recognized_text: str
    accuracy_score: float
    fluency_score: float
    completeness_score: float
    prosody_score: Optional[float] = None
    pronunciation_score: float
    words: List[dict] = []
    error_type: Optional[str] = None


class WordDetail(BaseModel):
    """단어별 상세 평가"""
    word: str
    accuracy_score: float
    error_type: str


@router.post("/assess", response_model=PronunciationResult)
async def assess_pronunciation(
    audio: UploadFile = File(..., description="음성 파일 (WAV 권장)"),
    reference_text: str = Form(..., description="읽어야 할 참조 텍스트"),
    enable_prosody: bool = Form(True, description="Prosody(억양/리듬) 평가 활성화")
):
    """
    발음 평가 테스트 엔드포인트

    - **audio**: WAV 형식의 음성 파일
    - **reference_text**: 사용자가 읽어야 할 영어 문장
    - **enable_prosody**: Prosody 평가 활성화 (en-US만 지원)

    반환값:
    - accuracy_score: 발음 정확도 (0-100)
    - fluency_score: 유창성 (0-100)
    - completeness_score: 완성도 (0-100)
    - prosody_score: 억양/리듬 점수 (0-100, en-US만)
    - pronunciation_score: 종합 점수 (0-100)
    - words: 단어별 상세 평가
    """

    # 임시 파일로 저장
    temp_file = None
    try:
        # 음성 파일 임시 저장
        suffix = ".wav" if audio.filename.endswith(".wav") else os.path.splitext(audio.filename)[1]
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        content = await audio.read()
        temp_file.write(content)
        temp_file.close()

        # Azure Speech 설정
        speech_config = speechsdk.SpeechConfig(
            subscription=settings.AZURE_SPEECH_KEY,
            region=settings.AZURE_SPEECH_REGION
        )
        audio_config = speechsdk.audio.AudioConfig(filename=temp_file.name)

        # 발음 평가 설정
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True
        )

        # Prosody 평가 활성화 (en-US만 지원)
        if enable_prosody:
            pronunciation_config.enable_prosody_assessment()

        # 인식기 생성
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            language="en-US",
            audio_config=audio_config
        )
        pronunciation_config.apply_to(speech_recognizer)

        # 발음 평가 실행
        result = speech_recognizer.recognize_once()

        # 결과 처리
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            assessment_result = speechsdk.PronunciationAssessmentResult(result)

            # JSON 결과에서 상세 정보 추출
            json_result = result.properties.get(
                speechsdk.PropertyId.SpeechServiceResponse_JsonResult
            )

            words_detail = []
            if json_result:
                try:
                    parsed = json.loads(json_result)
                    if "NBest" in parsed and len(parsed["NBest"]) > 0:
                        nbest = parsed["NBest"][0]
                        if "Words" in nbest:
                            for word_info in nbest["Words"]:
                                word_assessment = word_info.get("PronunciationAssessment", {})
                                words_detail.append({
                                    "word": word_info.get("Word", ""),
                                    "accuracy_score": word_assessment.get("AccuracyScore", 0),
                                    "error_type": word_assessment.get("ErrorType", "None")
                                })
                except json.JSONDecodeError:
                    pass

            return PronunciationResult(
                recognized_text=result.text,
                accuracy_score=assessment_result.accuracy_score,
                fluency_score=assessment_result.fluency_score,
                completeness_score=assessment_result.completeness_score,
                prosody_score=getattr(assessment_result, 'prosody_score', None),
                pronunciation_score=assessment_result.pronunciation_score,
                words=words_detail
            )

        elif result.reason == speechsdk.ResultReason.NoMatch:
            raise HTTPException(
                status_code=400,
                detail="음성을 인식할 수 없습니다. 더 명확하게 발음해주세요."
            )
        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            raise HTTPException(
                status_code=500,
                detail=f"평가 취소됨: {cancellation.reason}, {cancellation.error_details}"
            )
        else:
            raise HTTPException(
                status_code=500,
                detail=f"알 수 없는 오류: {result.reason}"
            )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"발음 평가 중 오류 발생: {str(e)}"
        )
    finally:
        # 임시 파일 삭제
        if temp_file and os.path.exists(temp_file.name):
            os.unlink(temp_file.name)


@router.get("/health")
async def health_check():
    """Azure Speech 연결 상태 확인"""
    try:
        # 설정 확인
        if not settings.AZURE_SPEECH_KEY or not settings.AZURE_SPEECH_REGION:
            return {
                "status": "error",
                "message": "Azure Speech 설정이 없습니다. AZURE_SPEECH_KEY와 AZURE_SPEECH_REGION을 확인하세요."
            }

        return {
            "status": "ok",
            "region": settings.AZURE_SPEECH_REGION,
            "key_configured": bool(settings.AZURE_SPEECH_KEY),
            "sdk_version": speechsdk.__version__
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }


@router.get("/sample-texts")
async def get_sample_texts():
    """테스트용 샘플 문장 목록"""
    return {
        "samples": [
            {
                "level": "easy",
                "text": "Hello, how are you today?",
                "korean": "안녕하세요, 오늘 어떠세요?"
            },
            {
                "level": "easy",
                "text": "Nice to meet you.",
                "korean": "만나서 반갑습니다."
            },
            {
                "level": "medium",
                "text": "I would like to schedule a meeting for tomorrow.",
                "korean": "내일 회의를 잡고 싶습니다."
            },
            {
                "level": "medium",
                "text": "Could you please send me the report by Friday?",
                "korean": "금요일까지 보고서를 보내주시겠어요?"
            },
            {
                "level": "hard",
                "text": "The quarterly financial analysis indicates a significant improvement in our revenue streams.",
                "korean": "분기별 재무 분석에 따르면 우리 수익원이 크게 개선되었습니다."
            },
            {
                "level": "hard",
                "text": "We need to restructure our organizational hierarchy to enhance operational efficiency.",
                "korean": "운영 효율성을 높이기 위해 조직 구조를 재편해야 합니다."
            }
        ]
    }
