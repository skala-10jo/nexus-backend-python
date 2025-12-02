"""
음성 번역 성능 측정 로거 (백엔드)

WebSocket 기반 실시간 음성 번역의 백엔드 성능을 측정합니다.

측정 항목:
- STT 인식 시간
- 번역 처리 시간
- 전체 처리 시간 (End-to-End)

사용 방법:
1. perf_logger.start_session(session_id)
2. perf_logger.start_timer(session_id, 'event_name')
3. perf_logger.end_timer(session_id, 'event_name', metadata)
4. perf_logger.end_session(session_id)
"""

import logging
import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PerformanceEvent:
    """성능 측정 이벤트"""
    event_name: str
    timestamp: float
    duration_ms: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


class PerformanceLogger:
    """백엔드 성능 측정 로거 (싱글톤)"""

    def __init__(self):
        self.sessions: Dict[str, Dict] = {}
        self.session_start_times: Dict[str, float] = {}
        self.timers: Dict[str, float] = {}

    def start_session(self, session_id: str):
        """
        새 세션 시작

        Args:
            session_id: 세션 ID
        """
        self.sessions[session_id] = []
        self.session_start_times[session_id] = time.time()
        logger.info(f"📊 Performance session started: {session_id}")

    def start_timer(self, session_id: str, event_name: str):
        """
        이벤트 타이머 시작

        Args:
            session_id: 세션 ID
            event_name: 이벤트 이름
        """
        key = f"{session_id}:{event_name}"
        self.timers[key] = time.time()

    def end_timer(
        self,
        session_id: str,
        event_name: str,
        metadata: Optional[Dict] = None
    ) -> Optional[float]:
        """
        이벤트 타이머 종료 및 기록

        Args:
            session_id: 세션 ID
            event_name: 이벤트 이름
            metadata: 추가 메타데이터

        Returns:
            측정된 시간 (ms)
        """
        key = f"{session_id}:{event_name}"
        start_time = self.timers.pop(key, None)

        if start_time is None:
            logger.warning(f"⚠️ Timer not found: {event_name}")
            return None

        duration_ms = (time.time() - start_time) * 1000
        self.record(session_id, event_name, duration_ms, metadata or {})
        return duration_ms

    def record(
        self,
        session_id: str,
        event_name: str,
        duration_ms: float,
        metadata: Optional[Dict] = None
    ):
        """
        이벤트 직접 기록 (타이머 없이)

        Args:
            session_id: 세션 ID
            event_name: 이벤트 이름
            duration_ms: 지속 시간 (ms)
            metadata: 메타데이터
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return

        event = PerformanceEvent(
            event_name=event_name,
            timestamp=time.time(),
            duration_ms=duration_ms,
            metadata=metadata or {}
        )

        self.sessions[session_id].append(event)
        logger.info(
            f"⏱️  {event_name}: {duration_ms:.2f}ms "
            f"{metadata if metadata else ''}"
        )

    def end_session(self, session_id: str):
        """
        세션 종료 및 통계 출력

        Args:
            session_id: 세션 ID
        """
        if session_id not in self.sessions:
            logger.warning(f"⚠️ Session not found: {session_id}")
            return

        stats = self.get_stats(session_id)
        self.print_stats(session_id, stats)

        # 세션 데이터 정리
        self.sessions.pop(session_id, None)
        self.session_start_times.pop(session_id, None)

    def get_stats(self, session_id: str) -> Dict:
        """
        세션 통계 계산

        Args:
            session_id: 세션 ID

        Returns:
            통계 데이터
        """
        if session_id not in self.sessions:
            return {"error": "Session not found"}

        events = self.sessions[session_id]

        if not events:
            logger.warning("⚠️ No events recorded")
            return {"error": "No events"}

        # 이벤트별 분류
        stt_events = [e for e in events if e.event_name == 'stt_recognition']
        translation_events = [e for e in events if e.event_name == 'translation']
        total_events = [e for e in events if e.event_name == 'total_processing']

        def calc_stats(event_list: List[PerformanceEvent]) -> Dict:
            """통계 계산 헬퍼"""
            if not event_list:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}

            durations = [e.duration_ms for e in event_list if e.duration_ms is not None]
            if not durations:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}

            return {
                "count": len(durations),
                "avg": sum(durations) / len(durations),
                "min": min(durations),
                "max": max(durations)
            }

        session_start = self.session_start_times.get(session_id, 0)
        session_duration = (time.time() - session_start) * 1000 if session_start else 0

        return {
            "session_id": session_id,
            "session_duration": session_duration,
            "total_events": len(events),
            "stt": calc_stats(stt_events),
            "translation": calc_stats(translation_events),
            "total_processing": calc_stats(total_events)
        }

    def print_stats(self, session_id: str, stats: Dict):
        """
        통계를 포맷팅해서 로그에 출력

        Args:
            session_id: 세션 ID
            stats: 통계 데이터
        """
        logger.info("\n" + "=" * 70)
        logger.info(f"🎯 Voice Translation Performance Stats - {session_id}")
        logger.info("=" * 70)
        logger.info(f"⏱️  Session Duration: {stats.get('session_duration', 0):.0f}ms")
        logger.info(f"📊 Total Events: {stats.get('total_events', 0)}")
        logger.info("")

        # STT 인식 시간
        if stats['stt']['count'] > 0:
            logger.info("🎤 STT Recognition Time:")
            logger.info(f"   Count: {stats['stt']['count']}")
            logger.info(f"   Avg:   {stats['stt']['avg']:.0f}ms")
            logger.info(f"   Min:   {stats['stt']['min']:.0f}ms")
            logger.info(f"   Max:   {stats['stt']['max']:.0f}ms")
            logger.info("")

        # 번역 시간
        if stats['translation']['count'] > 0:
            logger.info("🌐 Translation Time:")
            logger.info(f"   Count: {stats['translation']['count']}")
            logger.info(f"   Avg:   {stats['translation']['avg']:.0f}ms")
            logger.info(f"   Min:   {stats['translation']['min']:.0f}ms")
            logger.info(f"   Max:   {stats['translation']['max']:.0f}ms")
            logger.info("")

        # 전체 처리 시간
        if stats['total_processing']['count'] > 0:
            logger.info("⚡ Total Processing Time (End-to-End):")
            logger.info(f"   Count: {stats['total_processing']['count']}")
            logger.info(f"   Avg:   {stats['total_processing']['avg']:.0f}ms")
            logger.info(f"   Min:   {stats['total_processing']['min']:.0f}ms")
            logger.info(f"   Max:   {stats['total_processing']['max']:.0f}ms")
            logger.info("")

        logger.info("=" * 70 + "\n")


# 싱글톤 인스턴스
perf_logger = PerformanceLogger()
