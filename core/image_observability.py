"""Структурированные события и чистая агрегация метрик image-операций.

Логируемые события намеренно имеют allowlist полей: сюда нельзя передать
пользовательскую инструкцию, содержимое изображения, Telegram ID или секреты.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from statistics import median
from typing import Any

logger = logging.getLogger(__name__)

_ATTEMPT_FIELDS = (
    "operation_id",
    "attempt",
    "operation",
    "requested_model",
    "actual_model",
    "route_reason",
    "resolution",
    "reference_count",
    "duration_ms",
    "status",
    "error_code",
    "status_code",
    "cost",
    "cost_unknown",
)
_OUTCOME_FIELDS = (
    "operation_id",
    "task_id",
    "operation",
    "requested_model",
    "requested_quality",
    "actual_model",
    "quality",
    "route_reason",
    "delivered",
    "attempt_count",
    "fallback_used",
    "is_complaint_retry",
    "total_duration_ms",
    "width",
    "height",
    "bytes",
    "result_cost",
    "total_known_cost",
    "billing_unknown",
)
_SERVER_PROMOTION_REASONS = frozenset(
    {
        "invalid_quality_hint",
        "retry_after_fast",
        "retry_unknown_origin",
        "multi_reference_threshold",
        "quoted_text",
        "text_marker",
        "generated_result_improvement",
    }
)


def log_image_generation_attempt(event: Mapping[str, Any]) -> None:
    """Записывает безопасную структурированную запись одной API-попытки."""

    safe_event = {key: event.get(key) for key in _ATTEMPT_FIELDS if key in event}
    logger.info(
        "image_generation_attempt %s",
        json.dumps(safe_event, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )


def log_image_operation_summary(event: Mapping[str, Any]) -> None:
    """Записывает безопасный итог одной пользовательской image-операции."""

    safe_event = {key: event.get(key) for key in _OUTCOME_FIELDS if key in event}
    logger.info(
        "image_operation_summary %s",
        json.dumps(safe_event, ensure_ascii=False, separators=(",", ":"), sort_keys=True),
    )


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def aggregate_image_cost_report(
    operations: Iterable[Mapping[str, Any]],
    *,
    klein_cost: float | None = None,
    pro_cost: float | None = None,
    text_context_cost: float = 0.0,
) -> dict[str, Any]:
    """Агрегирует структурированные итоги image-операций без побочных эффектов.

    ``task_id`` объединяет исходную операцию и связанные жалобы/перегенерации;
    если он отсутствует, каждая запись считается отдельной задачей. Известная
    стоимость и неизвестные списания отражаются раздельно, поэтому отсутствие
    ``usage.cost`` никогда не интерпретируется как нулевая цена результата.
    Записи одной задачи передаются в хронологическом порядке: последний итог
    определяет, считается ли задача принятой.

    Ориентировочная цена стандартной задачи рассчитывается только при передаче
    известных ``klein_cost`` и ``pro_cost`` по формуле
    ``K + (f + u) * P + text_context_cost``. ``f`` — среднее число технических
    переключений, ``u`` — среднее число связанных перегенераций на задачу.
    """

    rows = [dict(operation) for operation in operations if isinstance(operation, Mapping)]
    count = len(rows)
    delivered_rows = [row for row in rows if row.get("delivered") is True]
    known_quality_rows = [
        row for row in rows if row.get("quality") in {"standard", "pro"}
    ]
    klein_count = sum(row.get("quality") == "standard" for row in known_quality_rows)

    promotion_counts: Counter[str] = Counter()
    for row in rows:
        reason = row.get("route_reason")
        if (
            row.get("quality") == "pro"
            and isinstance(reason, str)
            and reason in _SERVER_PROMOTION_REASONS
        ):
            promotion_counts[reason] += 1

    durations = [
        duration
        for row in delivered_rows
        if (duration := _nonnegative_number(row.get("total_duration_ms"))) is not None
    ]

    result_costs = [
        _nonnegative_number(row.get("result_cost")) for row in delivered_rows
    ]
    known_result_costs = [cost for cost in result_costs if cost is not None]
    unknown_delivered_results = sum(cost is None for cost in result_costs)

    known_operation_costs: dict[str, float] = defaultdict(float)
    task_unknown: dict[str, bool] = defaultdict(bool)
    task_accepted: dict[str, bool] = defaultdict(bool)
    task_fallbacks: Counter[str] = Counter()
    task_complaint_retries: Counter[str] = Counter()
    task_ids: set[str] = set()
    standard_route_task_ids: set[str] = set()
    unknown_operation_count = 0

    for index, row in enumerate(rows):
        operation_id = row.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id:
            operation_id = f"record-{index}"
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            task_id = operation_id
        task_ids.add(task_id)
        requested_quality = row.get("requested_quality")
        if requested_quality == "standard" or (
            requested_quality is None
            and (row.get("quality") == "standard" or row.get("fallback_used") is True)
        ):
            standard_route_task_ids.add(task_id)

        operation_cost = _nonnegative_number(row.get("total_known_cost"))
        if operation_cost is not None:
            known_operation_costs[task_id] += operation_cost
        if row.get("billing_unknown") is True:
            task_unknown[task_id] = True
            unknown_operation_count += 1
        # Строки операции должны быть переданы в хронологическом порядке:
        # последняя попытка/жалоба определяет состояние задачи.
        task_accepted[task_id] = row.get("delivered") is True
        if row.get("fallback_used") is True:
            task_fallbacks[task_id] += 1
        if row.get("is_complaint_retry") is True:
            task_complaint_retries[task_id] += 1

    accepted_task_ids = [task_id for task_id, accepted in task_accepted.items() if accepted]
    accepted_known_costs = [known_operation_costs[task_id] for task_id in accepted_task_ids]
    accepted_unknown_count = sum(task_unknown[task_id] for task_id in accepted_task_ids)
    accepted_count = len(accepted_task_ids)

    total_fallbacks = sum(task_fallbacks.values())
    total_complaint_retries = sum(task_complaint_retries.values())
    task_count = len(task_ids)
    standard_route_count = len(standard_route_task_ids)
    formula_k = _nonnegative_number(klein_cost)
    formula_p = _nonnegative_number(pro_cost)
    context_cost = _nonnegative_number(text_context_cost)
    if context_cost is None:
        context_cost = 0.0
    standard_route_fallbacks = sum(
        task_fallbacks[task_id] for task_id in standard_route_task_ids
    )
    standard_route_retries = sum(
        task_complaint_retries[task_id] for task_id in standard_route_task_ids
    )
    fallback_per_task = (
        standard_route_fallbacks / standard_route_count if standard_route_count else None
    )
    retries_per_task = (
        standard_route_retries / standard_route_count if standard_route_count else None
    )
    estimated_cost_per_task = None
    if (
        formula_k is not None
        and formula_p is not None
        and fallback_per_task is not None
        and retries_per_task is not None
    ):
        estimated_cost_per_task = (
            formula_k
            + (fallback_per_task + retries_per_task) * formula_p
            + context_cost
        )

    return {
        "operation_count": count,
        "delivered_count": len(delivered_rows),
        "klein_share": (
            klein_count / len(known_quality_rows) if known_quality_rows else None
        ),
        "klein_share_of_all_operations": klein_count / count if count else None,
        "klein_count": klein_count,
        "known_quality_count": len(known_quality_rows),
        "quality_unknown_count": count - len(known_quality_rows),
        "server_promotions_by_reason": {
            reason: {
                "count": reason_count,
                "share": reason_count / count if count else None,
                "promotion_share": (
                    reason_count / sum(promotion_counts.values())
                    if promotion_counts
                    else None
                ),
            }
            for reason, reason_count in sorted(promotion_counts.items())
        },
        "technical_fallback_share": (
            sum(row.get("fallback_used") is True for row in rows) / count
            if count
            else None
        ),
        "complaint_retry_share": (
            sum(row.get("is_complaint_retry") is True for row in rows) / count
            if count
            else None
        ),
        "duration_ms": {
            "median": median(durations) if durations else None,
            "p95": _percentile(durations, 0.95),
            "sample_count": len(durations),
        },
        "delivered_result_cost": {
            "known_total": math.fsum(known_result_costs),
            "known_average": (
                math.fsum(known_result_costs) / len(known_result_costs)
                if known_result_costs
                else None
            ),
            "unknown_count": unknown_delivered_results,
            "unknown_share": (
                unknown_delivered_results / len(delivered_rows)
                if delivered_rows
                else None
            ),
        },
        "all_operations_cost": {
            "known_total": math.fsum(
                _nonnegative_number(row.get("total_known_cost")) or 0.0
                for row in rows
            ),
            "unknown_operation_count": unknown_operation_count,
            "unknown_operation_share": unknown_operation_count / count if count else None,
        },
        "accepted_task_cost": {
            "accepted_task_count": accepted_count,
            "known_subtotal_total": math.fsum(accepted_known_costs),
            "known_subtotal_average": (
                math.fsum(accepted_known_costs) / accepted_count
                if accepted_count
                else None
            ),
            "unknown_task_count": accepted_unknown_count,
            "unknown_task_share": (
                accepted_unknown_count / accepted_count if accepted_count else None
            ),
        },
        "estimated_cost_formula": {
            "K": formula_k,
            "P": formula_p,
            "f": fallback_per_task,
            "u": retries_per_task,
            "sample_task_count": standard_route_count,
            "text_context_cost": context_cost,
            "per_task": estimated_cost_per_task,
        },
    }
