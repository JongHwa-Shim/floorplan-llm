"""
JSONL 평면도 데이터 로더 모듈.

여러 평면도가 연달아 저장된 JSONL 파일에서 특정 plan_id의 데이터를 로드한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import orjson


class FloorplanLoader:
    """JSONL 파일에서 평면도 데이터를 로드하는 클래스.

    JSONL 파일은 한 줄에 하나의 평면도 JSON이 저장된 형식이며,
    여러 파일에 걸쳐 있을 수 있다.

    Attributes:
        jsonl_paths: 로드 대상 JSONL 파일 경로 리스트.
    """

    def __init__(self, jsonl_paths: list[Path]) -> None:
        """초기화.

        Args:
            jsonl_paths: JSONL 파일 경로 리스트.
        """
        self.jsonl_paths = jsonl_paths

    @classmethod
    def from_directory(cls, directory: Path, pattern: str = "*.jsonl") -> "FloorplanLoader":
        """디렉토리에서 JSONL 파일들을 자동으로 수집해 로더를 생성한다.

        Args:
            directory: JSONL 파일이 저장된 디렉토리 경로.
            pattern: 파일 검색 패턴 (기본값: "*.jsonl").

        Returns:
            FloorplanLoader 인스턴스.

        Raises:
            FileNotFoundError: 디렉토리가 존재하지 않는 경우.
        """
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"디렉토리를 찾을 수 없습니다: {directory}")

        # single_output.jsonl을 항상 최우선으로 검색 (테스트 시 신규 추출 결과 우선 반영)
        def _sort_key(p: Path) -> tuple:
            return (0, p.name) if p.name == "single_output.jsonl" else (1, p.name)

        paths = sorted(directory.glob(pattern), key=_sort_key)
        return cls(paths)

    def load_by_plan_id(self, plan_id: str) -> Optional[dict]:
        """특정 plan_id에 해당하는 평면도 데이터를 반환한다.

        모든 JSONL 파일을 순서대로 탐색하며 첫 번째 일치 항목을 반환한다.

        Args:
            plan_id: 찾을 평면도의 ID 문자열.

        Returns:
            평면도 데이터 딕셔너리. 찾지 못하면 None.
        """
        for path in self.jsonl_paths:
            result = self._search_in_file(path, plan_id)
            if result is not None:
                return result
        return None

    def load_all(self) -> list[dict]:
        """모든 JSONL 파일의 평면도 데이터를 로드한다.

        Args:
            없음.

        Returns:
            전체 평면도 데이터 딕셔너리 리스트.
        """
        results = []
        for path in self.jsonl_paths:
            results.extend(self._load_file(path))
        return results

    def get_all_plan_ids(self) -> list[str]:
        """모든 JSONL 파일에서 plan_id 목록을 수집한다.

        Args:
            없음.

        Returns:
            plan_id 문자열 리스트.
        """
        ids = []
        for path in self.jsonl_paths:
            for record in self._load_file(path):
                ids.append(record["plan_id"])
        return ids

    def _search_in_file(self, path: Path, plan_id: str) -> Optional[dict]:
        """단일 JSONL 파일에서 plan_id를 탐색한다.

        Args:
            path: 탐색할 JSONL 파일 경로.
            plan_id: 찾을 plan_id.

        Returns:
            일치하는 평면도 딕셔너리. 없으면 None.
        """
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = orjson.loads(line)
                if str(record.get("plan_id")) == str(plan_id):
                    return record
        return None

    def _load_file(self, path: Path) -> list[dict]:
        """단일 JSONL 파일의 모든 레코드를 로드한다.

        Args:
            path: 로드할 JSONL 파일 경로.

        Returns:
            레코드 딕셔너리 리스트.
        """
        records = []
        with open(path, "rb") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                records.append(orjson.loads(line))
        return records
