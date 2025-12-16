"""
Speaking Tutor Service for business logic.
Coordinates between API layer and Agent layer.
"""
import os
import uuid
import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified
from sqlalchemy import desc

from app.config import settings
from app.models.speaking_tutor import SpeakingAnalysisSession, SpeakingUtterance
from agent.speaking_tutor import DiarizationAgent, SpeakingFeedbackAgent, MeetingSummaryAgent

logger = logging.getLogger(__name__)


class SpeakingTutorService:
    """
    Service class for Speaking Tutor functionality.
    Handles file upload, analysis orchestration, and data retrieval.
    """

    # Supported audio file extensions
    ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
    MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

    def __init__(self):
        self.upload_dir = os.path.join(settings.upload_dir, "speaking_tutor")
        os.makedirs(self.upload_dir, exist_ok=True)

        # Initialize agents (lazy initialization for diarization/feedback)
        self.diarization_agent = None
        self.feedback_agent = None
        self.summary_agent = MeetingSummaryAgent()

    async def upload_audio(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        language: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Upload and save audio file, create analysis session.

        Args:
            file_content: Audio file bytes
            filename: Original filename
            user_id: User UUID string
            language: Language code (e.g., en-US)
            db: Database session

        Returns:
            Dict with session_id, status, and message

        Raises:
            ValueError: If file validation fails
        """
        # Validate file extension
        ext = os.path.splitext(filename)[1].lower()
        if ext not in self.ALLOWED_EXTENSIONS:
            raise ValueError(f"Unsupported file format. Allowed: {', '.join(self.ALLOWED_EXTENSIONS)}")

        # Validate file size
        if len(file_content) > self.MAX_FILE_SIZE:
            raise ValueError(f"File too large. Maximum size: {self.MAX_FILE_SIZE // (1024*1024)}MB")

        # Generate unique filename
        session_id = uuid.uuid4()
        safe_filename = f"{session_id}{ext}"
        file_path = os.path.join(self.upload_dir, safe_filename)

        # Save file
        try:
            with open(file_path, "wb") as f:
                f.write(file_content)
            logger.info(f"Audio file saved: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save audio file: {str(e)}")
            raise ValueError(f"Failed to save file: {str(e)}")

        # Create session record
        session = SpeakingAnalysisSession(
            id=session_id,
            user_id=uuid.UUID(user_id),
            original_filename=filename,
            file_path=file_path,
            file_size=len(file_content),
            language=language,
            status="PENDING"
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        logger.info(f"✅ Analysis session created: {session.id}")

        # Trigger async analysis in background
        asyncio.create_task(self._run_analysis(str(session.id), file_path, language, user_id))

        return {
            "session_id": str(session.id),
            "status": "PROCESSING",
            "message": "파일이 업로드되었습니다. 분석을 시작합니다."
        }

    async def _run_analysis(
        self,
        session_id: str,
        file_path: str,
        language: str,
        user_id: str
    ) -> None:
        """
        Run async analysis using DiarizationAgent.
        Updates session status and creates utterance records.
        """
        from app.database import SessionLocal

        db = SessionLocal()
        try:
            session = db.query(SpeakingAnalysisSession).filter(
                SpeakingAnalysisSession.id == uuid.UUID(session_id)
            ).first()

            if not session:
                logger.error(f"Session not found: {session_id}")
                return

            # Update status to PROCESSING
            session.status = "PROCESSING"
            session.progress = 10
            db.commit()
            logger.info(f"🎙️ Starting analysis for session: {session_id}")

            def progress_callback(progress: int, message: str):
                """Update progress in database."""
                try:
                    session.progress = progress
                    db.commit()
                    logger.debug(f"Progress: {progress}% - {message}")
                except Exception as e:
                    logger.warning(f"Failed to update progress: {e}")

            # Initialize DiarizationAgent
            if self.diarization_agent is None:
                self.diarization_agent = DiarizationAgent()

            # Run diarization
            result = await self.diarization_agent.process(
                audio_file_path=file_path,
                language=language,
                progress_callback=progress_callback
            )

            # Save utterances
            utterances_data = result.get("utterances", [])
            for utt_data in utterances_data:
                utterance = SpeakingUtterance(
                    session_id=session.id,
                    speaker_id=utt_data["speaker_id"],
                    text=utt_data["text"],
                    start_time_ms=utt_data["start_time_ms"],
                    end_time_ms=utt_data["end_time_ms"],
                    confidence=utt_data.get("confidence", 0.9),
                    sequence_number=utt_data["sequence_number"]
                )
                db.add(utterance)

            # Update session with results
            session.status = "COMPLETED"
            session.progress = 100
            session.speaker_count = result.get("speaker_count", 1)
            session.utterance_count = len(utterances_data)
            session.duration_seconds = result.get("duration_seconds", 0.0)
            session.completed_at = datetime.now(timezone.utc)

            # Initialize speaker labels
            speaker_labels = {}
            for i in range(1, session.speaker_count + 1):
                speaker_labels[str(i)] = f"화자 {i}"
            session.speaker_labels = speaker_labels

            # Generate AI summary of meeting content
            try:
                summary = await self._generate_meeting_summary(utterances_data, language)
                session.summary = summary
                logger.info(f"Summary generated: {summary[:50]}...")
            except Exception as summary_err:
                logger.warning(f"Summary generation failed: {summary_err}")
                # Fallback: use first utterance as summary
                if utterances_data:
                    first_text = utterances_data[0].get("text", "")
                    session.summary = first_text[:100] + "..." if len(first_text) > 100 else first_text

            db.commit()
            logger.info(f"Analysis completed: {session_id}, {len(utterances_data)} utterances")

        except Exception as e:
            logger.error(f"Analysis failed for {session_id}: {str(e)}")
            try:
                session = db.query(SpeakingAnalysisSession).filter(
                    SpeakingAnalysisSession.id == uuid.UUID(session_id)
                ).first()
                if session:
                    session.status = "FAILED"
                    session.error_message = str(e)
                    session.progress = 0
                    db.commit()
            except Exception as db_err:
                logger.error(f"Failed to update error status: {db_err}")
        finally:
            db.close()

    def get_analysis_result(
        self,
        session_id: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """
        Get analysis result for a session.

        Args:
            session_id: Session UUID string
            user_id: User UUID string
            db: Database session

        Returns:
            Analysis result with status and utterances (if completed)

        Raises:
            ValueError: If session not found or access denied
        """
        session = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.id == uuid.UUID(session_id),
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).first()

        if not session:
            raise ValueError("Session not found or access denied")

        if session.status in ["PENDING", "PROCESSING"]:
            return {
                "session_id": str(session.id),
                "status": session.status,
                "progress": session.progress,
                "message": self._get_progress_message(session.status, session.progress)
            }

        if session.status == "FAILED":
            return {
                "session_id": str(session.id),
                "status": "FAILED",
                "progress": 0,
                "message": session.error_message or "Analysis failed"
            }

        # COMPLETED - return full result
        utterances = db.query(SpeakingUtterance).filter(
            SpeakingUtterance.session_id == session.id
        ).order_by(SpeakingUtterance.sequence_number).all()

        # Build speaker info
        speaker_stats = {}
        for utt in utterances:
            if utt.speaker_id not in speaker_stats:
                speaker_stats[utt.speaker_id] = 0
            speaker_stats[utt.speaker_id] += 1

        speakers = [
            {
                "id": sid,
                "label": session.speaker_labels.get(str(sid), f"화자 {sid}"),
                "utteranceCount": count
            }
            for sid, count in sorted(speaker_stats.items())
        ]

        utterance_list = [
            {
                "id": str(utt.id),
                "speakerId": utt.speaker_id,
                "speakerLabel": session.speaker_labels.get(str(utt.speaker_id), f"화자 {utt.speaker_id}"),
                "text": utt.text,
                "startTimeMs": utt.start_time_ms,
                "endTimeMs": utt.end_time_ms,
                "confidence": utt.confidence,
                "hasFeedback": utt.feedback is not None,
                "feedback": self._convert_feedback(utt.feedback) if utt.feedback else None,
                "sequenceNumber": utt.sequence_number
            }
            for utt in utterances
        ]

        return {
            "session_id": str(session.id),
            "status": "COMPLETED",
            "durationSeconds": session.duration_seconds,
            "speakerCount": session.speaker_count,
            "utteranceCount": session.utterance_count,
            "speakers": speakers,
            "utterances": utterance_list
        }

    def _get_progress_message(self, status: str, progress: int) -> str:
        """Get human-readable progress message."""
        if status == "PENDING":
            return "분석 대기 중..."
        if progress < 20:
            return "오디오 파일 처리 중..."
        if progress < 50:
            return "음성 인식 중..."
        if progress < 80:
            return "화자 분리 중..."
        return "결과 정리 중..."

    def _convert_feedback(self, feedback: Dict) -> Dict:
        """Convert feedback from DB format to API format."""
        return {
            "grammarCorrections": feedback.get("grammar_corrections", []),
            "suggestions": feedback.get("suggestions", []),
            "improvedSentence": feedback.get("improved_sentence", ""),
            "score": feedback.get("score", 0),
            "scoreBreakdown": feedback.get("score_breakdown")
        }

    async def _generate_meeting_summary(
        self,
        utterances_data: List[Dict],
        language: str
    ) -> str:
        """
        Generate a brief summary of meeting content using MeetingSummaryAgent.

        Args:
            utterances_data: List of utterance dictionaries with 'text' and 'speaker_id'
            language: Language code (e.g., 'en-US', 'ko-KR')

        Returns:
            Brief summary string (max ~100 chars for card display)
        """
        return await self.summary_agent.process(utterances_data, language)

    async def generate_feedback(
        self,
        utterance_id: str,
        user_id: str,
        context: Optional[str],
        db: Session
    ) -> Dict[str, Any]:
        """
        Generate feedback for an utterance using SpeakingFeedbackAgent.
        """
        # Verify access and get session info
        utterance = db.query(SpeakingUtterance).join(SpeakingAnalysisSession).filter(
            SpeakingUtterance.id == uuid.UUID(utterance_id),
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).first()

        if not utterance:
            raise ValueError("Utterance not found or access denied")

        # Get session for language info
        session = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.id == utterance.session_id
        ).first()

        language = "en"
        if session and session.language:
            # Convert "en-US" to "en"
            language = session.language.split("-")[0]

        logger.info(f"Generating feedback for utterance: {utterance_id}")

        try:
            # Initialize FeedbackAgent if needed
            if self.feedback_agent is None:
                self.feedback_agent = SpeakingFeedbackAgent()

            # Generate feedback using AI
            feedback = await self.feedback_agent.process(
                utterance_text=utterance.text,
                context=context,
                language=language
            )

            logger.info(f"Feedback generated: score={feedback.get('score', 0)}")

        except Exception as e:
            logger.error(f"Feedback generation failed: {str(e)}")
            # Return fallback feedback on error
            feedback = {
                "grammar_corrections": [f"피드백 생성 중 오류가 발생했습니다: {str(e)}"],
                "suggestions": ["다시 시도해 주세요."],
                "improved_sentence": utterance.text,
                "score": 0,
                "score_breakdown": {
                    "grammar": 0,
                    "vocabulary": 0,
                    "fluency": 0,
                    "clarity": 0
                }
            }

        # Save feedback to DB
        utterance.feedback = feedback
        db.commit()

        return {
            "utterance_id": str(utterance.id),
            "feedback": self._convert_feedback(feedback)
        }

    def update_speaker_label(
        self,
        session_id: str,
        speaker_id: int,
        label: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Update speaker label for a session."""
        session = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.id == uuid.UUID(session_id),
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).first()

        if not session:
            raise ValueError("Session not found or access denied")

        # Update speaker labels
        labels = dict(session.speaker_labels or {})
        labels[str(speaker_id)] = label
        session.speaker_labels = labels
        flag_modified(session, "speaker_labels")
        db.commit()

        return {
            "speaker_id": speaker_id,
            "label": label,
            "updated": True
        }

    def get_learning_data(
        self,
        session_id: str,
        user_id: str,
        speaker_ids: Optional[List[int]],
        db: Session
    ) -> Dict[str, Any]:
        """Get learning mode data for a session."""
        session = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.id == uuid.UUID(session_id),
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).first()

        if not session:
            raise ValueError("Session not found or access denied")

        if session.status != "COMPLETED":
            raise ValueError("Analysis not completed yet")

        # Get utterances with feedback
        query = db.query(SpeakingUtterance).filter(
            SpeakingUtterance.session_id == session.id,
            SpeakingUtterance.feedback.isnot(None)
        )

        # Filter by multiple speaker IDs
        if speaker_ids is not None and len(speaker_ids) > 0:
            query = query.filter(SpeakingUtterance.speaker_id.in_(speaker_ids))

        utterances = query.order_by(SpeakingUtterance.sequence_number).all()

        learning_items = []
        for utt in utterances:
            if utt.feedback:
                improved = utt.feedback.get("improved_sentence", utt.text)
                if improved != utt.text:  # Only include if there's improvement
                    learning_items.append({
                        "utteranceId": str(utt.id),
                        "originalText": utt.text,
                        "improvedText": improved,
                        "grammarCorrections": utt.feedback.get("grammar_corrections", []),
                        "suggestions": utt.feedback.get("suggestions", []),
                        "score": utt.feedback.get("score", 0),
                        "scoreBreakdown": utt.feedback.get("score_breakdown", {}),
                        "speakerId": utt.speaker_id,
                        "practiceCount": 0
                    })

        return {
            "session_id": str(session.id),
            "learningItems": learning_items
        }

    def get_session_list(
        self,
        user_id: str,
        page: int,
        size: int,
        db: Session
    ) -> Dict[str, Any]:
        """Get paginated list of user's sessions."""
        query = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).order_by(desc(SpeakingAnalysisSession.created_at))

        total = query.count()
        sessions = query.offset(page * size).limit(size).all()

        return {
            "sessions": [
                {
                    "id": str(s.id),
                    "originalFilename": s.original_filename,
                    "status": s.status,
                    "speakerCount": s.speaker_count,
                    "utteranceCount": s.utterance_count,
                    "durationSeconds": s.duration_seconds,
                    "language": s.language,
                    "summary": s.summary,
                    "createdAt": s.created_at.isoformat()
                }
                for s in sessions
            ],
            "total": total
        }

    def delete_session(
        self,
        session_id: str,
        user_id: str,
        db: Session
    ) -> Dict[str, Any]:
        """Delete a session and its file."""
        session = db.query(SpeakingAnalysisSession).filter(
            SpeakingAnalysisSession.id == uuid.UUID(session_id),
            SpeakingAnalysisSession.user_id == uuid.UUID(user_id)
        ).first()

        if not session:
            raise ValueError("Session not found or access denied")

        # Delete file
        if os.path.exists(session.file_path):
            try:
                os.remove(session.file_path)
                logger.info(f"Deleted audio file: {session.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete file: {str(e)}")

        # Delete session (cascade deletes utterances)
        db.delete(session)
        db.commit()

        logger.info(f"Deleted session: {session_id}")

        return {
            "session_id": session_id,
            "deleted": True,
            "message": "Session deleted successfully"
        }
