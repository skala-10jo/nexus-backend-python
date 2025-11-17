"""
화자 구분 Agent (Speaker Diarization)

Phase 1: 간단한 에너지 기반 화자 구분
- 음성 에너지 레벨 변화 감지
- 발화 패턴 분석
- 최대 5명의 화자 지원

Phase 2 (향후): pyannote.audio 통합
- 딥러닝 기반 화자 임베딩
- 높은 정확도의 화자 클러스터링
"""
from typing import Optional, Dict, List
from collections import deque
import time


class SpeakerDiarizationAgent:
    """
    화자 구분 Agent

    음성 에너지 레벨과 발화 패턴을 분석하여 여러 화자를 구분합니다.
    BaseAgent를 상속하지 않음 (OpenAI API 사용하지 않음)
    """

    def __init__(self):
        # 화자 히스토리 (최근 100개 발화)
        self.speaker_history: deque = deque(maxlen=100)

        # 현재 활성 화자
        self.current_speaker_id = 1

        # 화자별 에너지 프로필
        self.speaker_profiles: Dict[int, Dict] = {}

        # 설정
        self.max_speakers = 5  # 최대 화자 수
        self.energy_threshold = 0.3  # 에너지 변화 임계값
        self.silence_threshold = 2.0  # 무음 임계값 (초)
        self.min_speaker_duration = 1.0  # 최소 발화 시간 (초)

        # 마지막 발화 시간
        self.last_speech_time = time.time()

    async def process(
        self,
        audio_energy: float,
        timestamp: float,
        duration: float = 1.0
    ) -> Dict[str, any]:
        """
        오디오 에너지와 타임스탬프를 분석하여 화자 ID 추정

        Args:
            audio_energy: 오디오 에너지 레벨 (0.0 ~ 1.0)
            timestamp: 타임스탬프 (Unix time)
            duration: 오디오 청크 길이 (초)

        Returns:
            {
                "speaker_id": 1,  # 화자 ID (1~5)
                "confidence": 0.85,  # 신뢰도
                "is_new_speaker": False  # 화자 전환 여부
            }
        """
        current_time = timestamp

        # 1. 무음 감지 (에너지가 매우 낮음)
        if audio_energy < 0.1:
            return {
                "speaker_id": self.current_speaker_id,
                "confidence": 0.0,
                "is_new_speaker": False
            }

        # 2. 무음 구간 후 새로운 발화 감지
        time_since_last_speech = current_time - self.last_speech_time
        is_after_silence = time_since_last_speech > self.silence_threshold

        # 3. 화자 전환 감지
        is_new_speaker = False

        if not self.speaker_history:
            # 첫 번째 발화
            is_new_speaker = True

        elif is_after_silence:
            # 긴 무음 후 새로운 화자 가능성
            is_new_speaker = self._should_switch_speaker(
                audio_energy,
                current_time
            )

        else:
            # 에너지 레벨 변화로 화자 전환 감지
            last_energy = self.speaker_history[-1]['energy']
            energy_diff = abs(audio_energy - last_energy)

            if energy_diff > self.energy_threshold:
                is_new_speaker = self._should_switch_speaker(
                    audio_energy,
                    current_time
                )

        # 4. 화자 ID 할당
        if is_new_speaker:
            self.current_speaker_id = self._assign_speaker_id(audio_energy)

        # 5. 화자 프로필 업데이트
        self._update_speaker_profile(
            self.current_speaker_id,
            audio_energy,
            duration
        )

        # 6. 히스토리 업데이트
        self.speaker_history.append({
            "speaker_id": self.current_speaker_id,
            "energy": audio_energy,
            "timestamp": current_time,
            "duration": duration
        })

        self.last_speech_time = current_time

        # 7. 신뢰도 계산
        confidence = self._calculate_confidence(
            self.current_speaker_id,
            audio_energy
        )

        return {
            "speaker_id": self.current_speaker_id,
            "confidence": confidence,
            "is_new_speaker": is_new_speaker
        }

    def _should_switch_speaker(
        self,
        audio_energy: float,
        current_time: float
    ) -> bool:
        """화자 전환이 필요한지 판단"""

        if not self.speaker_history:
            return True

        # 최근 발화들의 평균 에너지
        recent_speeches = [
            s for s in list(self.speaker_history)[-5:]
            if s['speaker_id'] == self.current_speaker_id
        ]

        if not recent_speeches:
            return True

        avg_energy = sum(s['energy'] for s in recent_speeches) / len(recent_speeches)

        # 에너지 차이가 크면 화자 전환
        energy_diff = abs(audio_energy - avg_energy)

        if energy_diff > self.energy_threshold:
            return True

        # 일정 시간 이상 같은 화자가 말하면 전환 가능성 높음
        last_speaker_duration = current_time - recent_speeches[0]['timestamp']

        if last_speaker_duration > 10.0:  # 10초 이상
            return energy_diff > self.energy_threshold * 0.5

        return False

    def _assign_speaker_id(self, audio_energy: float) -> int:
        """새로운 발화에 화자 ID 할당"""

        # 1. 기존 화자 프로필과 매칭
        best_match_id = None
        best_match_score = float('inf')

        for speaker_id, profile in self.speaker_profiles.items():
            # 에너지 레벨 유사도
            energy_diff = abs(audio_energy - profile['avg_energy'])

            if energy_diff < best_match_score:
                best_match_score = energy_diff
                best_match_id = speaker_id

        # 2. 충분히 유사한 화자가 있으면 재사용
        if best_match_id and best_match_score < 0.2:
            return best_match_id

        # 3. 새로운 화자 ID 할당
        existing_ids = set(self.speaker_profiles.keys())
        for candidate_id in range(1, self.max_speakers + 1):
            if candidate_id not in existing_ids:
                return candidate_id

        # 4. 최대 화자 수 도달 시 순환
        return (self.current_speaker_id % self.max_speakers) + 1

    def _update_speaker_profile(
        self,
        speaker_id: int,
        audio_energy: float,
        duration: float
    ):
        """화자 프로필 업데이트 (에너지 통계)"""

        if speaker_id not in self.speaker_profiles:
            self.speaker_profiles[speaker_id] = {
                "avg_energy": audio_energy,
                "total_duration": duration,
                "speech_count": 1,
                "energy_history": [audio_energy]
            }
        else:
            profile = self.speaker_profiles[speaker_id]

            # 에너지 히스토리 업데이트 (최근 20개만 유지)
            profile['energy_history'].append(audio_energy)
            if len(profile['energy_history']) > 20:
                profile['energy_history'].pop(0)

            # 평균 에너지 재계산
            profile['avg_energy'] = sum(profile['energy_history']) / len(profile['energy_history'])

            # 발화 통계 업데이트
            profile['total_duration'] += duration
            profile['speech_count'] += 1

    def _calculate_confidence(
        self,
        speaker_id: int,
        audio_energy: float
    ) -> float:
        """화자 구분 신뢰도 계산 (0.0 ~ 1.0)"""

        if speaker_id not in self.speaker_profiles:
            # 새로운 화자는 낮은 신뢰도
            return 0.5

        profile = self.speaker_profiles[speaker_id]

        # 1. 에너지 레벨 일관성
        energy_diff = abs(audio_energy - profile['avg_energy'])
        energy_confidence = max(0.0, 1.0 - energy_diff / 0.5)

        # 2. 발화 횟수 (많을수록 신뢰도 높음)
        count_confidence = min(1.0, profile['speech_count'] / 10.0)

        # 3. 종합 신뢰도
        confidence = (energy_confidence * 0.7 + count_confidence * 0.3)

        return round(confidence, 2)

    def get_speaker_stats(self) -> Dict[int, Dict]:
        """
        모든 화자의 통계 정보 반환

        Returns:
            {
                1: {
                    "avg_energy": 0.65,
                    "total_duration": 15.5,
                    "speech_count": 8
                },
                2: {...}
            }
        """
        return {
            speaker_id: {
                "avg_energy": round(profile['avg_energy'], 2),
                "total_duration": round(profile['total_duration'], 2),
                "speech_count": profile['speech_count']
            }
            for speaker_id, profile in self.speaker_profiles.items()
        }

    def reset(self):
        """모든 화자 데이터 초기화"""
        self.speaker_history.clear()
        self.speaker_profiles.clear()
        self.current_speaker_id = 1
        self.last_speech_time = time.time()

    def get_current_speaker(self) -> int:
        """현재 활성 화자 ID 반환"""
        return self.current_speaker_id

    def get_speaker_count(self) -> int:
        """감지된 화자 수 반환"""
        return len(self.speaker_profiles)
