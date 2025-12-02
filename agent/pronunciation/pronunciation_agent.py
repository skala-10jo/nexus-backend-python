"""
Azure Pronunciation Assessment Agent

Azure Speech Service의 Pronunciation Assessment 기능을 사용하여
발음, 유창성, 강세, 호흡 등을 평가하는 Agent입니다.

Features:
- 음소(Phoneme) 단위 발음 평가
- 단어(Word) 단위 발음 평가
- 유창성(Fluency) 평가
- 완성도(Completeness) 평가
- 운율(Prosody) 평가 (강세, 억양)
"""
import logging
import asyncio
from typing import Dict, Any, Optional, List
import azure.cognitiveservices.speech as speechsdk
from agent.base_agent import BaseAgent
from app.core.azure_speech_token_manager import AzureSpeechTokenManager as AzureSpeechAgent

logger = logging.getLogger(__name__)


class PronunciationAssessmentAgent(BaseAgent):
    """
    Azure Speech Pronunciation Assessment Agent (싱글톤)

    발음 평가를 수행하는 Agent입니다.

    Features:
    - 음소 단위 정확도 평가
    - 단어 단위 정확도 평가
    - 유창성, 완성도, 운율 평가
    - 실시간 스트리밍 지원
    - 싱글톤 패턴

    Example:
        >>> agent = PronunciationAssessmentAgent.get_instance()
        >>> result = await agent.assess_pronunciation(
        ...     audio_data=audio_bytes,
        ...     reference_text="Hello, how are you?",
        ...     language="en-US"
        ... )
        >>> print(f"Accuracy: {result['accuracy_score']}")
        >>> print(f"Fluency: {result['fluency_score']}")
    """

    _instance: Optional['PronunciationAssessmentAgent'] = None

    def __init__(self):
        """
        Initialize Pronunciation Assessment Agent.

        Note: 직접 호출하지 말고 get_instance()를 사용하세요.
        """
        super().__init__()
        self.speech_agent = AzureSpeechAgent.get_instance()
        logger.info("PronunciationAssessmentAgent initialized")

    async def process(self, *args, **kwargs):
        """
        BaseAgent의 추상 메서드 구현.

        PronunciationAssessmentAgent는 assess_pronunciation()을 직접 사용하므로
        이 메서드는 사용하지 않지만, BaseAgent 인터페이스를 만족하기 위해 구현합니다.
        """
        return await self.assess_pronunciation(*args, **kwargs)

    @classmethod
    def get_instance(cls) -> 'PronunciationAssessmentAgent':
        """
        싱글톤 인스턴스 반환

        Returns:
            PronunciationAssessmentAgent 싱글톤 인스턴스
        """
        if cls._instance is None:
            cls._instance = cls()
            logger.info("Created new PronunciationAssessmentAgent singleton instance")
        return cls._instance

    def _convert_to_wav(self, audio_data: bytes) -> bytes:
        """
        WebM/기타 오디오 형식을 WAV (16kHz, mono, 16-bit PCM)로 변환

        Args:
            audio_data: 원본 오디오 데이터

        Returns:
            bytes: WAV 형식 오디오 데이터
        """
        import io
        import subprocess
        import tempfile

        try:
            # 임시 파일에 원본 오디오 저장
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as input_file:
                input_file.write(audio_data)
                input_path = input_file.name

            # WAV 출력 경로
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as output_file:
                output_path = output_file.name

            # ffmpeg로 변환 (16kHz, mono, 16-bit PCM)
            subprocess.run([
                'ffmpeg',
                '-i', input_path,
                '-ar', '16000',  # 16kHz sampling rate (Azure 권장)
                '-ac', '1',      # mono
                '-sample_fmt', 's16',  # 16-bit PCM
                '-y',  # overwrite
                output_path
            ], check=True, capture_output=True)

            # WAV 파일 읽기
            with open(output_path, 'rb') as f:
                wav_data = f.read()

            # 임시 파일 삭제
            import os
            os.unlink(input_path)
            os.unlink(output_path)

            logger.info(f"Audio converted: {len(audio_data)} bytes → {len(wav_data)} bytes WAV")
            return wav_data

        except subprocess.CalledProcessError as e:
            logger.error(f"ffmpeg conversion failed: {e.stderr.decode() if e.stderr else str(e)}")
            # 변환 실패시 원본 데이터 반환 (이미 WAV일 수 있음)
            return audio_data
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            return audio_data

    async def assess_pronunciation(
        self,
        audio_data: bytes,
        reference_text: str,
        language: str = "en-US",
        granularity: str = "Phoneme"  # "Phoneme", "Word", "FullText"
    ) -> Dict[str, Any]:
        """
        발음 평가 수행

        Args:
            audio_data: 오디오 데이터 (WAV/WebM/기타 형식 자동 변환)
            reference_text: 평가할 기준 텍스트 (사용자가 읽어야 할 텍스트)
            language: BCP-47 언어 코드 (예: en-US, en-GB)
            granularity: 평가 세분화 수준 (Phoneme/Word/FullText)

        Returns:
            Dict[str, Any]: {
                "accuracy_score": float,        # 발음 정확도 (0-100)
                "fluency_score": float,          # 유창성 점수 (0-100)
                "completeness_score": float,     # 완성도 점수 (0-100)
                "prosody_score": float,          # 운율 점수 (0-100)
                "pronunciation_score": float,    # 전체 발음 점수 (0-100)
                "words": List[Dict],             # 단어별 상세 평가
                "recognized_text": str           # 인식된 텍스트
            }

        Raises:
            Exception: 발음 평가 실패 시
        """
        try:
            logger.info(
                f"Starting pronunciation assessment: "
                f"language={language}, "
                f"granularity={granularity}, "
                f"reference_text='{reference_text[:50]}...'"
            )

            # WebM을 WAV로 변환
            audio_data = self._convert_to_wav(audio_data)

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성 (토큰 기반)
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

            # Pronunciation Assessment Config 생성
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=getattr(
                    speechsdk.PronunciationAssessmentGranularity,
                    granularity
                ),
                enable_miscue=True  # 잘못 읽은 단어 감지
            )

            # Prosody(운율) 평가 활성화
            pronunciation_config.enable_prosody_assessment()

            # PushAudioInputStream 생성
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Speech Recognizer 생성
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Pronunciation Assessment 적용
            pronunciation_config.apply_to(recognizer)

            # 오디오 데이터 푸시
            push_stream.write(audio_data)
            push_stream.close()

            # 음성 인식 실행 (비동기)
            result = await asyncio.to_thread(recognizer.recognize_once)

            # 결과 처리
            if result.reason == speechsdk.ResultReason.RecognizedSpeech:
                # Pronunciation Assessment 결과 추출
                pronunciation_result = speechsdk.PronunciationAssessmentResult(result)

                # 전체 점수 (None인 경우 0.0으로 처리)
                response = {
                    "accuracy_score": pronunciation_result.accuracy_score or 0.0,
                    "fluency_score": pronunciation_result.fluency_score or 0.0,
                    "completeness_score": pronunciation_result.completeness_score or 0.0,
                    "prosody_score": pronunciation_result.prosody_score or 0.0,
                    "pronunciation_score": pronunciation_result.pronunciation_score or 0.0,
                    "recognized_text": result.text,
                    "reference_text": reference_text,
                    "words": []
                }

                # 단어별 상세 평가 (JSON 파싱)
                import json
                details = json.loads(result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                ))

                # NBest 결과에서 단어별 평가 추출
                if "NBest" in details and len(details["NBest"]) > 0:
                    best_result = details["NBest"][0]

                    if "Words" in best_result:
                        for word_data in best_result["Words"]:
                            pronunciation_assessment = word_data.get("PronunciationAssessment", {})

                            word_info = {
                                "word": word_data.get("Word", ""),
                                "accuracy_score": pronunciation_assessment.get("AccuracyScore", 0),
                                "error_type": pronunciation_assessment.get("ErrorType", "None"),
                                # Prosody 관련 세부 정보 추가
                                "prosody": {
                                    "break_score": pronunciation_assessment.get("Feedback", {}).get("Prosody", {}).get("Break", {}).get("BreakScore", None),
                                    "intonation_score": pronunciation_assessment.get("Feedback", {}).get("Prosody", {}).get("Intonation", {}).get("IntonationScore", None)
                                },
                                "phonemes": []
                            }

                            # 음소별 평가
                            if "Phonemes" in word_data:
                                for phoneme_data in word_data["Phonemes"]:
                                    phoneme_info = {
                                        "phoneme": phoneme_data.get("Phoneme", ""),
                                        "accuracy_score": phoneme_data.get("PronunciationAssessment", {}).get("AccuracyScore", 0),
                                        "NBestPhonemes": phoneme_data.get("PronunciationAssessment", {}).get("NBestPhonemes", [])
                                    }
                                    word_info["phonemes"].append(phoneme_info)

                            response["words"].append(word_info)

                logger.info(
                    f"Pronunciation assessment success: "
                    f"accuracy={response['accuracy_score']:.1f}, "
                    f"fluency={response['fluency_score']:.1f}, "
                    f"prosody={response['prosody_score']:.1f}"
                )

                return response

            elif result.reason == speechsdk.ResultReason.NoMatch:
                logger.warning("No speech recognized in audio for pronunciation assessment")
                return {
                    "accuracy_score": 0.0,
                    "fluency_score": 0.0,
                    "completeness_score": 0.0,
                    "prosody_score": 0.0,
                    "pronunciation_score": 0.0,
                    "recognized_text": "",
                    "reference_text": reference_text,
                    "words": [],
                    "error": "No speech recognized"
                }

            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                error_msg = f"Pronunciation assessment canceled: {cancellation.reason}, {cancellation.error_details}"
                logger.error(error_msg)
                raise Exception(error_msg)

            else:
                error_msg = f"Unexpected pronunciation assessment result reason: {result.reason}"
                logger.error(error_msg)
                raise Exception(error_msg)

        except Exception as e:
            logger.error(f"Pronunciation assessment failed: {str(e)}", exc_info=True)
            raise Exception(f"발음 평가 실패: {str(e)}")

    async def assess_pronunciation_stream(
        self,
        reference_text: str,
        language: str = "en-US",
        granularity: str = "Phoneme"
    ) -> tuple:
        """
        실시간 스트리밍 발음 평가를 위한 Recognizer 및 PushStream 반환

        WebSocket에서 사용하기 위한 스트리밍 인터페이스입니다.

        Args:
            reference_text: 평가할 기준 텍스트
            language: BCP-47 언어 코드
            granularity: 평가 세분화 수준 (Phoneme/Word/FullText)

        Returns:
            tuple: (recognizer, push_stream) - 호출자가 직접 관리

        Example:
            >>> recognizer, push_stream = await agent.assess_pronunciation_stream(
            ...     reference_text="Hello world",
            ...     language="en-US"
            ... )
            >>> # WebSocket에서 오디오 청크 수신 시
            >>> push_stream.write(audio_chunk)
            >>> # 종료 시
            >>> push_stream.close()
            >>> recognizer.stop_continuous_recognition()
        """
        try:
            logger.info(
                f"Setting up streaming pronunciation assessment: "
                f"language={language}, "
                f"granularity={granularity}, "
                f"reference_text='{reference_text[:50]}...'"
            )

            # Azure Speech 토큰 가져오기
            token, region = await self.speech_agent.get_token()

            # Speech Config 생성
            speech_config = speechsdk.SpeechConfig(
                subscription=None,
                region=region,
                auth_token=token
            )
            speech_config.speech_recognition_language = language

            # Pronunciation Assessment Config 생성
            pronunciation_config = speechsdk.PronunciationAssessmentConfig(
                reference_text=reference_text,
                grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
                granularity=getattr(
                    speechsdk.PronunciationAssessmentGranularity,
                    granularity
                ),
                enable_miscue=True
            )

            # Prosody 평가 활성화
            pronunciation_config.enable_prosody_assessment()

            # PushAudioInputStream 생성
            push_stream = speechsdk.audio.PushAudioInputStream()
            audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

            # Speech Recognizer 생성
            recognizer = speechsdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config
            )

            # Pronunciation Assessment 적용
            pronunciation_config.apply_to(recognizer)

            logger.info("Streaming pronunciation assessment setup complete")
            return recognizer, push_stream

        except Exception as e:
            logger.error(f"Streaming pronunciation assessment setup failed: {str(e)}", exc_info=True)
            raise Exception(f"스트리밍 발음 평가 설정 실패: {str(e)}")


# 싱글톤 인스턴스 생성 함수
def get_pronunciation_agent() -> PronunciationAssessmentAgent:
    """
    Pronunciation Assessment Agent 싱글톤 인스턴스 반환

    Returns:
        PronunciationAssessmentAgent: 싱글톤 인스턴스
    """
    return PronunciationAssessmentAgent.get_instance()
