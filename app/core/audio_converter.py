"""
오디오 형식 변환 유틸리티

WebM, OGG 등을 Azure Speech SDK가 지원하는 WAV로 변환
"""
import io
import logging
from pydub import AudioSegment

logger = logging.getLogger(__name__)


def convert_to_wav(audio_data: bytes, source_format: str = "webm") -> bytes:
    """
    오디오 데이터를 WAV 형식으로 변환

    Args:
        audio_data: 원본 오디오 데이터
        source_format: 원본 형식 (webm, ogg, mp3 등)

    Returns:
        bytes: WAV 형식의 오디오 데이터 (16kHz, mono, 16-bit PCM)

    Raises:
        Exception: 변환 실패 시
    """
    try:
        logger.info(f"Converting audio from {source_format} to WAV")

        # BytesIO로 래핑
        audio_file = io.BytesIO(audio_data)

        # AudioSegment로 로드 (자동으로 형식 감지)
        audio = AudioSegment.from_file(audio_file, format=source_format)

        # Azure Speech SDK 요구사항에 맞게 변환
        # - 샘플레이트: 16kHz
        # - 채널: Mono
        # - 샘플 폭: 16-bit
        audio = audio.set_frame_rate(16000)
        audio = audio.set_channels(1)
        audio = audio.set_sample_width(2)  # 16-bit = 2 bytes

        # WAV로 export
        output = io.BytesIO()
        audio.export(output, format="wav")
        wav_data = output.getvalue()

        logger.info(f"Audio conversion successful: {len(audio_data)} bytes -> {len(wav_data)} bytes")

        return wav_data

    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}", exc_info=True)
        raise Exception(f"오디오 변환 실패: {str(e)}")


def detect_audio_format(filename: str) -> str:
    """
    파일명에서 오디오 형식 추출

    Args:
        filename: 파일명 (예: recording.webm)

    Returns:
        str: 오디오 형식 (webm, ogg, mp3 등)
    """
    if not filename:
        return "webm"  # 기본값

    extension = filename.lower().split('.')[-1]

    # 확장자 매핑
    format_map = {
        'webm': 'webm',
        'ogg': 'ogg',
        'mp3': 'mp3',
        'wav': 'wav',
        'm4a': 'mp4',  # pydub에서는 m4a를 mp4로 처리
        'mp4': 'mp4'
    }

    return format_map.get(extension, 'webm')
