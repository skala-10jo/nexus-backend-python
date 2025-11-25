"""
Azure Speech 실시간 스트리밍 발음 평가 Agent

WebSocket을 통해 오디오 청크를 실시간으로 받아 발음을 평가합니다.
"""
import json
from typing import Dict, Optional
import azure.cognitiveservices.speech as speechsdk
from agent.voice.base_azure_agent import BaseAzureAgent


class PronunciationStreamingAgent(BaseAzureAgent):
    """
    실시간 스트리밍 발음 평가 Agent

    Azure Speech Service의 PushAudioInputStream을 사용하여
    실시간으로 오디오 청크를 받아 발음을 평가합니다.
    """

    def __init__(self, speech_key: str, speech_region: str):
        """
        Args:
            speech_key: Azure Speech Service API Key
            speech_region: Azure 리전 (예: koreacentral)
        """
        self.speech_key = speech_key
        self.speech_region = speech_region

    def create_recognizer(
        self,
        reference_text: str,
        language: str = "en-US"
    ) -> tuple:
        """
        스트리밍용 Speech Recognizer 생성

        Args:
            reference_text: 평가할 기준 텍스트
            language: 언어 코드

        Returns:
            (recognizer, push_stream) 튜플
        """
        # 1. Azure Speech Config
        speech_config = speechsdk.SpeechConfig(
            subscription=self.speech_key,
            region=self.speech_region
        )
        speech_config.speech_recognition_language = language

        # 2. Push Audio Stream 생성
        push_stream = speechsdk.audio.PushAudioInputStream()
        audio_config = speechsdk.audio.AudioConfig(stream=push_stream)

        # 3. Pronunciation Assessment Config (음소 레벨로 설정)
        pronunciation_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True
        )
        # Prosody(억양/리듬) 평가 활성화 (en-US만 지원)
        if language == "en-US":
            pronunciation_config.enable_prosody_assessment()

        # 4. Speech Recognizer
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config
        )
        pronunciation_config.apply_to(recognizer)

        return recognizer, push_stream

    def process(self, result) -> Optional[Dict]:
        """
        발음 평가 결과 처리 (BaseAzureAgent 인터페이스 구현)

        Args:
            result: Azure Speech Recognition 결과

        Returns:
            발음 평가 결과 딕셔너리 또는 None

        Raises:
            Exception: 결과 파싱 실패 시
        """
        return self.parse_result(result)

    def parse_result(self, result) -> Optional[Dict]:
        """
        인식 결과 파싱

        Args:
            result: Azure Speech Recognition 결과

        Returns:
            발음 평가 결과 또는 None
        """
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pronunciation_result = json.loads(
                result.properties.get(
                    speechsdk.PropertyId.SpeechServiceResponse_JsonResult
                )
            )

            nbest = pronunciation_result.get("NBest", [{}])[0]
            assessment = nbest.get("PronunciationAssessment", {})
            words = nbest.get("Words", [])

            word_details = []
            for word in words:
                word_assessment = word.get("PronunciationAssessment", {})

                # 음소별 점수 파싱
                phonemes = word.get("Phonemes", [])
                phoneme_details = []
                for phoneme in phonemes:
                    phoneme_assessment = phoneme.get("PronunciationAssessment", {})
                    phoneme_details.append({
                        "phoneme": phoneme.get("Phoneme", ""),
                        "accuracy_score": phoneme_assessment.get("AccuracyScore", 0.0)
                    })

                word_details.append({
                    "word": word.get("Word", ""),
                    "accuracy_score": word_assessment.get("AccuracyScore", 0.0),
                    "error_type": word_assessment.get("ErrorType", "None"),
                    "phonemes": phoneme_details
                })

            return {
                "pronunciation_score": assessment.get("PronScore", 0.0),
                "accuracy_score": assessment.get("AccuracyScore", 0.0),
                "fluency_score": assessment.get("FluencyScore", 0.0),
                "completeness_score": assessment.get("CompletenessScore", 0.0),
                "prosody_score": assessment.get("ProsodyScore"),  # en-US만 지원
                "recognized_text": nbest.get("Display", ""),
                "words": word_details
            }

        elif result.reason == speechsdk.ResultReason.NoMatch:
            return {
                "pronunciation_score": 0.0,
                "accuracy_score": 0.0,
                "fluency_score": 0.0,
                "completeness_score": 0.0,
                "recognized_text": "",
                "words": [],
                "error": "음성을 인식할 수 없습니다"
            }

        elif result.reason == speechsdk.ResultReason.Canceled:
            cancellation = result.cancellation_details
            error_msg = f"발음 평가 실패: {cancellation.reason}"
            if cancellation.reason == speechsdk.CancellationReason.Error:
                error_msg += f" - {cancellation.error_details}"
            return {
                "error": error_msg
            }

        return None
