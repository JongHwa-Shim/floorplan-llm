"""여러 관계를 동시에 만족하는 일대일 방 대응의 존재 여부를 검사한다."""

from __future__ import annotations


def has_consistent_assignment(candidates: dict[int, list[int]], constraints: list[tuple]) -> bool:
    """입력 RID마다 서로 다른 출력 방을 대응시켜 모든 관계를 검사한다.

    Mod Record: 조건별로 같은 RID를 다른 출력 방에 대응시키지 않도록 한다.
    후보 축소, 이분 매칭 가능성 검사, 최소 후보 우선 탐색으로 가지치기한다.
    기하 계산은 호출부에서 한 번 수행하고 허용 인덱스 쌍만 전달한다.

    Args:
        candidates: 입력 RID별 출력 방 후보.
        constraints: (입력 RID A, 입력 RID B, 허용 출력 인덱스 쌍 집합) 목록.
    Returns:
        모든 조건을 동시에 만족하는 일대일 대응이 있으면 True.
    Raises:
        ValueError: 조건이 후보 목록에 없는 RID를 참조할 때.
    """
    if any(a not in candidates or b not in candidates for a, b, _ in constraints):
        raise ValueError("관계에 사용된 RID의 후보가 없습니다.")
    failed = set()

    def injective_possible(domains):
        """후보만으로 서로 다른 방을 배정할 수 있는지 검사한다.

        Args:
            domains: 현재 RID별 후보 집합.
        Returns:
            전체 RID를 덮는 이분 매칭의 존재 여부.
        Raises:
            없음.
        """
        owners = {}

        def augment(rid, visited):
            """증대 경로 하나를 찾아 출력 방을 배정한다.

            Args:
                rid: 배정할 입력 방.
                visited: 이번 탐색에서 방문한 출력 방.
            Returns:
                배정 성공 여부.
            Raises:
                없음.
            """
            for out in domains[rid]:
                if out in visited:
                    continue
                visited.add(out)
                if out not in owners or augment(owners[out], visited):
                    owners[out] = rid
                    return True
            return False

        return all(augment(rid, set()) for rid in sorted(domains, key=lambda r: len(domains[r])))

    def solve(domains):
        """후보를 전파한 뒤 필요한 경우에만 분기한다.

        Args:
            domains: 현재 RID별 후보 집합.
        Returns:
            관계를 만족하는 일대일 대응의 존재 여부.
        Raises:
            없음.
        """
        changed = True
        while changed:
            if any(not d for d in domains.values()):
                return False
            changed = False
            singles = [next(iter(d)) for d in domains.values() if len(d) == 1]
            if len(singles) != len(set(singles)):
                return False
            for rid, domain in list(domains.items()):
                if len(domain) > 1:
                    reduced = domain - set(singles)
                    if reduced != domain:
                        domains[rid] = reduced
                        changed = True
            for a, b, allowed in constraints:
                pairs = [(x, y) for x, y in allowed
                         if x != y and x in domains[a] and y in domains[b] and a != b]
                first, second = {x for x, _ in pairs}, {y for _, y in pairs}
                if first != domains[a] or second != domains[b]:
                    domains[a], domains[b] = first, second
                    changed = True
        if not injective_possible(domains):
            return False
        key = tuple((rid, tuple(sorted(d))) for rid, d in sorted(domains.items()))
        if key in failed:
            return False
        undecided = [rid for rid, d in domains.items() if len(d) > 1]
        if not undecided:
            return True
        rid = min(undecided, key=lambda r: (len(domains[r]), r))
        for out in sorted(domains[rid]):
            if solve({r: ({out} if r == rid else set(d)) for r, d in domains.items()}):
                return True
        failed.add(key)
        return False

    return solve({rid: set(values) for rid, values in candidates.items()})
