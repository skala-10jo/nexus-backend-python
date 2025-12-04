"""
Speaking Tutor API endpoints for audio analysis and feedback.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile, Form
from sqlalchemy.orm import Session
from typing import Optional
import logging

from app.database import get_db
from app.auth import get_current_user
from app.schemas.speaking_tutor import (
    UploadResponse,
    AnalysisProgressResponse,
    AnalysisCompleteResponse,
    FeedbackRequest,
    FeedbackResponse,
    UpdateSpeakerLabelRequest,
    UpdateSpeakerLabelResponse,
    LearningModeResponse,
    SessionListResponse,
    DeleteSessionResponse
)
from app.services.speaking_tutor_service import SpeakingTutorService

logger = logging.getLogger(__name__)

router = APIRouter()
speaking_tutor_service = SpeakingTutorService()


@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    language: str = Form(default="en-US"),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Upload audio file for speaker diarization and analysis.

    **Supported formats**: WAV, MP3, M4A, OGG, FLAC
    **Max size**: 100MB

    Returns session ID to poll for analysis results.
    """
    try:
        user_id = str(user["user_id"])
        logger.info(f"🎵 Audio upload: user={user_id}, file={file.filename}, language={language}")

        # Read file content
        file_content = await file.read()

        result = await speaking_tutor_service.upload_audio(
            file_content=file_content,
            filename=file.filename,
            user_id=user_id,
            language=language,
            db=db
        )

        return UploadResponse(
            sessionId=result["session_id"],
            status=result["status"],
            message=result["message"]
        )

    except ValueError as e:
        logger.error(f"❌ Upload validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/analysis/{session_id}")
async def get_analysis(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get analysis status or results for a session.

    Poll this endpoint until status is COMPLETED or FAILED.

    Returns:
    - PENDING/PROCESSING: Progress information
    - COMPLETED: Full analysis with utterances
    - FAILED: Error message
    """
    try:
        user_id = str(user["user_id"])
        result = speaking_tutor_service.get_analysis_result(
            session_id=session_id,
            user_id=user_id,
            db=db
        )

        # Return appropriate response based on status
        if result["status"] in ["PENDING", "PROCESSING", "FAILED"]:
            return AnalysisProgressResponse(
                sessionId=result["session_id"],
                status=result["status"],
                progress=result.get("progress", 0),
                message=result["message"]
            )
        else:
            return result  # Full response for COMPLETED

    except ValueError as e:
        logger.error(f"❌ Analysis fetch error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Analysis fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch analysis: {str(e)}")


@router.post("/feedback", response_model=FeedbackResponse)
async def generate_feedback(
    request: FeedbackRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Generate AI feedback for a specific utterance.

    Uses GPT-4o to analyze the utterance and provide:
    - Grammar corrections
    - Improvement suggestions
    - Improved sentence
    - Score breakdown
    """
    try:
        user_id = str(user["user_id"])
        logger.info(f"📝 Feedback request: user={user_id}, utterance={request.utteranceId}")

        result = await speaking_tutor_service.generate_feedback(
            utterance_id=request.utteranceId,
            user_id=user_id,
            context=request.context,
            db=db
        )

        return FeedbackResponse(
            utteranceId=result["utterance_id"],
            feedback=result["feedback"]
        )

    except ValueError as e:
        logger.error(f"❌ Feedback error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Feedback generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback generation failed: {str(e)}")


@router.patch("/speaker/{session_id}/{speaker_id}", response_model=UpdateSpeakerLabelResponse)
async def update_speaker_label(
    session_id: str,
    speaker_id: int,
    request: UpdateSpeakerLabelRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update the label for a speaker in a session.

    Example: Change "화자 1" to "나" or "김과장"
    """
    try:
        user_id = str(user["user_id"])
        logger.info(f"🏷️ Update speaker label: session={session_id}, speaker={speaker_id}, label={request.label}")

        result = speaking_tutor_service.update_speaker_label(
            session_id=session_id,
            speaker_id=speaker_id,
            label=request.label,
            user_id=user_id,
            db=db
        )

        return UpdateSpeakerLabelResponse(
            speakerId=result["speaker_id"],
            label=result["label"],
            updated=result["updated"]
        )

    except ValueError as e:
        logger.error(f"❌ Speaker label update error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Speaker label update failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update speaker label: {str(e)}")


@router.get("/learning/{session_id}", response_model=LearningModeResponse)
async def get_learning_data(
    session_id: str,
    speaker_id: Optional[int] = Query(default=None, alias="speakerId"),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get learning mode data for a session.

    Returns utterances that have improvement suggestions,
    formatted for practice/learning mode.

    Optionally filter by speaker ID.
    """
    try:
        user_id = str(user["user_id"])
        logger.info(f"📚 Learning data request: session={session_id}, speaker={speaker_id}")

        result = speaking_tutor_service.get_learning_data(
            session_id=session_id,
            user_id=user_id,
            speaker_id=speaker_id,
            db=db
        )

        return LearningModeResponse(
            sessionId=result["session_id"],
            learningItems=result["learningItems"]
        )

    except ValueError as e:
        logger.error(f"❌ Learning data error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Learning data fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch learning data: {str(e)}")


@router.get("/sessions", response_model=SessionListResponse)
async def get_sessions(
    page: int = Query(default=0, ge=0),
    size: int = Query(default=10, ge=1, le=50),
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get paginated list of user's analysis sessions.

    Use for history view.
    """
    try:
        user_id = str(user["user_id"])

        result = speaking_tutor_service.get_session_list(
            user_id=user_id,
            page=page,
            size=size,
            db=db
        )

        return SessionListResponse(
            sessions=result["sessions"],
            total=result["total"]
        )

    except Exception as e:
        logger.error(f"❌ Session list fetch failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to fetch sessions: {str(e)}")


@router.delete("/session/{session_id}", response_model=DeleteSessionResponse)
async def delete_session(
    session_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete an analysis session and its audio file.

    This action is irreversible.
    """
    try:
        user_id = str(user["user_id"])
        logger.info(f"🗑️ Delete session request: session={session_id}")

        result = speaking_tutor_service.delete_session(
            session_id=session_id,
            user_id=user_id,
            db=db
        )

        return DeleteSessionResponse(
            sessionId=result["session_id"],
            deleted=result["deleted"],
            message=result["message"]
        )

    except ValueError as e:
        logger.error(f"❌ Delete session error: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"❌ Delete session failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
