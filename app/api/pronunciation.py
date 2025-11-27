"""
Pronunciation Assessment API 엔드포인트

Azure Speech Service를 사용한 발음 평가 API
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import logging
import base64
from agent.pronunciation.pronunciation_agent import PronunciationAssessmentAgent
from app.schemas.pronunciation import (
    PronunciationAssessmentRequest,
    PronunciationAssessmentResponse,
    PronunciationFeedbackSummary
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/pronunciation", tags=["Pronunciation Assessment"])


@router.options("/assess")
async def options_pronunciation_assess():
    """브라우저 preflight용 OPTIONS 허용"""
    return JSONResponse(content=None, status_code=200)


@router.post(
    "/assess",
    response_model=dict,
    summary="발음 평가",
    description="""
    사용자의 음성을 평가하여 발음, 유창성, 강세 등에 대한 상세 피드백을 제공합니다.

    - 음소(Phoneme) 단위 정확도 평가
    - 단어(Word) 단위 정확도 평가
    - 유창성(Fluency) 평가
    - 완성도(Completeness) 평가
    - 운율(Prosody) 평가 (강세, 억양)

    **사용 시나리오:**
    1. 사용자가 주어진 텍스트를 읽음
    2. 음성을 Base64로 인코딩하여 전송
    3. Azure Speech Service가 발음 평가 수행
    4. 단어/음소 단위 상세 피드백 반환

    **응답 시간:** 1-3초 (오디오 길이에 따라)
    """
)
async def assess_pronunciation(
    request: PronunciationAssessmentRequest,
    http_request: Request
):
    """
    발음 평가 수행

    Request Body:
        {
            "audio_data": "Base64 인코딩된 오디오",
            "reference_text": "Hello, how are you?",
            "language": "en-US",
            "granularity": "Phoneme"
        }

    Returns:
        {
            "success": true,
            "message": "Pronunciation assessment completed",
            "data": {
                "accuracy_score": 85.5,
                "fluency_score": 90.0,
                "completeness_score": 100.0,
                "prosody_score": 88.0,
                "pronunciation_score": 87.2,
                "recognized_text": "Hello, how are you?",
                "reference_text": "Hello, how are you?",
                "words": [...]
            }
        }

    Raises:
        HTTPException: 발음 평가 실패 시
    """
    try:
        logger.info(
            f"Pronunciation assessment request: "
            f"language={request.language}, "
            f"granularity={request.granularity}, "
            f"reference_text='{request.reference_text[:50]}...'"
        )

        # Base64 디코딩
        try:
            audio_bytes = base64.b64decode(request.audio_data)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid Base64 audio data: {str(e)}"
            )

        # 오디오 크기 제한 (10MB)
        if len(audio_bytes) > 10 * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail="Audio data too large (max 10MB)"
            )

        # 싱글톤 Agent 인스턴스 가져오기
        agent = PronunciationAssessmentAgent.get_instance()

        # 발음 평가 수행
        result = await agent.assess_pronunciation(
            audio_data=audio_bytes,
            reference_text=request.reference_text,
            language=request.language,
            granularity=request.granularity
        )

        # Pydantic 모델로 변환
        response_data = PronunciationAssessmentResponse(**result)

        # JSON 응답
        resp = JSONResponse(
            content={
                "success": True,
                "message": "Pronunciation assessment completed successfully",
                "data": response_data.model_dump()
            }
        )

        # CORS 헤더 추가
        origin = http_request.headers.get("origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"

        logger.info(
            f"Pronunciation assessment success: "
            f"score={result['pronunciation_score']:.1f}, "
            f"accuracy={result['accuracy_score']:.1f}"
        )

        return resp

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pronunciation assessment failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Pronunciation assessment failed: {str(e)}"
        )


@router.post(
    "/feedback",
    response_model=dict,
    summary="발음 피드백 요약",
    description="""
    발음 평가 결과를 분석하여 사용자 친화적인 피드백을 생성합니다.

    - 문제가 있는 단어 식별 (정확도 70% 이하)
    - 문제가 있는 음소 식별 (정확도 70% 이하)
    - 전반적인 피드백 메시지 생성

    **사용 시나리오:**
    1. `/assess` 엔드포인트에서 발음 평가 수행
    2. 평가 결과를 이 엔드포인트로 전송
    3. 사용자 친화적인 피드백 메시지 받기
    """
)
async def generate_feedback(
    assessment_result: PronunciationAssessmentResponse,
    http_request: Request
):
    """
    발음 평가 결과를 기반으로 피드백 생성

    Request Body:
        PronunciationAssessmentResponse 객체 (assess 엔드포인트 응답)

    Returns:
        {
            "success": true,
            "message": "Feedback generated",
            "data": {
                "overall_score": 87.2,
                "accuracy": 85.5,
                "fluency": 90.0,
                "prosody": 88.0,
                "problem_words": ["world"],
                "problem_phonemes": [...],
                "feedback_message": "..."
            }
        }
    """
    try:
        logger.info("Generating pronunciation feedback")

        # 문제 단어 식별 (정확도 70% 이하)
        problem_words = [
            word.word
            for word in assessment_result.words
            if word.accuracy_score < 70.0
        ]

        # 문제 음소 식별
        problem_phonemes = []
        for word in assessment_result.words:
            for phoneme in word.phonemes:
                if phoneme.accuracy_score < 70.0:
                    problem_phonemes.append({
                        "word": word.word,
                        "phoneme": phoneme.phoneme,
                        "accuracy": phoneme.accuracy_score
                    })

        # 피드백 메시지 생성
        overall_score = assessment_result.pronunciation_score

        if overall_score >= 90:
            feedback_message = "Excellent pronunciation! Your speech is clear and natural."
        elif overall_score >= 80:
            feedback_message = "Good pronunciation overall. "
            if problem_words:
                feedback_message += f"Consider practicing these words: {', '.join(problem_words[:3])}."
        elif overall_score >= 70:
            feedback_message = "Fair pronunciation. "
            if problem_words:
                feedback_message += f"Focus on improving: {', '.join(problem_words[:3])}. "
            if problem_phonemes:
                feedback_message += f"Pay attention to the sounds: {', '.join([p['phoneme'] for p in problem_phonemes[:3]])}."
        else:
            feedback_message = "Keep practicing! "
            if problem_words:
                feedback_message += f"Start with these words: {', '.join(problem_words[:3])}. "
            feedback_message += "Consider working with a pronunciation coach or using more practice materials."

        # 유창성 피드백 추가
        if assessment_result.fluency_score < 70:
            feedback_message += " Try to speak more smoothly and naturally."

        # 운율 피드백 추가
        if assessment_result.prosody_score < 70:
            feedback_message += " Work on your stress and intonation patterns."

        # 응답 생성
        feedback = PronunciationFeedbackSummary(
            overall_score=overall_score,
            accuracy=assessment_result.accuracy_score,
            fluency=assessment_result.fluency_score,
            prosody=assessment_result.prosody_score,
            problem_words=problem_words,
            problem_phonemes=problem_phonemes,
            feedback_message=feedback_message
        )

        # JSON 응답
        resp = JSONResponse(
            content={
                "success": True,
                "message": "Feedback generated successfully",
                "data": feedback.model_dump()
            }
        )

        # CORS 헤더 추가
        origin = http_request.headers.get("origin")
        if origin:
            resp.headers["Access-Control-Allow-Origin"] = origin
        else:
            resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"

        logger.info("Feedback generated successfully")
        return resp

    except Exception as e:
        logger.error(f"Feedback generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Feedback generation failed: {str(e)}"
        )


@router.get(
    "/languages",
    summary="지원 언어 목록 조회",
    description="발음 평가를 지원하는 언어 목록을 반환합니다."
)
async def get_supported_languages(http_request: Request):
    """
    발음 평가 지원 언어 목록

    Returns:
        {
            "success": true,
            "message": "Supported languages retrieved",
            "data": {
                "languages": [
                    {"code": "en-US", "name": "English (US)"},
                    {"code": "en-GB", "name": "English (UK)"},
                    ...
                ]
            }
        }
    """
    # 현재 지원하는 언어 목록
    supported_languages = [
        {"code": "en-US", "name": "English (US)"},
        {"code": "en-GB", "name": "English (UK)"},
        {"code": "en-AU", "name": "English (Australia)"},
        {"code": "en-CA", "name": "English (Canada)"},
        {"code": "en-IN", "name": "English (India)"},
    ]

    resp = JSONResponse(
        content={
            "success": True,
            "message": "Supported languages retrieved",
            "data": {
                "languages": supported_languages
            }
        }
    )

    # CORS 헤더 추가
    origin = http_request.headers.get("origin")
    if origin:
        resp.headers["Access-Control-Allow-Origin"] = origin
    else:
        resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "*"

    return resp
