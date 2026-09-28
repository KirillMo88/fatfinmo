from __future__ import annotations

from typing import Any


PATTERN_RU = {
    "IMPULSE": "обычный импульс",
    "IMPULSE_TRUNCATED_5": "кандидат импульса с усечённой пятой",
    "ZIGZAG": "зигзаг A–B–C",
    "FLAT_REGULAR": "обычная плоская коррекция",
    "FLAT_EXPANDED": "расширенная плоская коррекция",
    "TRIANGLE_CONTRACTING": "сужающийся треугольник",
    "TRIANGLE_BARRIER": "барьерный треугольник",
    "DOUBLE_ZIGZAG": "двойной зигзаг W–X–Y",
    "DOUBLE_THREE": "двойная тройка W–X–Y",
    "ENDING_DIAGONAL_CONTRACTING_33333": "сужающаяся конечная диагональ",
}


def build_summary(
    snapshot: dict[str, Any],
    scenario_id: str | None,
    focus_node_id: str | None = None,
    visible_degree: str = "Auto",
) -> str:
    scenarios = {row.get("scenario_id"): row for row in snapshot.get("scenarios", [])}
    scenario = scenarios.get(scenario_id or snapshot.get("main_scenario_id"))
    if not scenario:
        reasons = snapshot.get("unresolved_reasons") or ["допустимый счёт пока не выделен"]
        return f"На выбранной степени однозначный счёт пока не выделен. {str(reasons[0]).replace('UNRESOLVED:', '').strip()}"
    nodes = {row.get("node_id"): row for row in snapshot.get("nodes", [])}
    node = nodes.get(focus_node_id or scenario.get("root_node_id"))
    if not node:
        return "Сценарий сохранён, но его фокусный узел недоступен в этом снимке."

    pattern = PATTERN_RU.get(node.get("pattern_type"), str(node.get("pattern_type", "структура")))
    labels = node.get("labels") or []
    last_label = labels[-1].get("label") if labels else "последний участок"
    endpoint_status = node.get("endpoint_status")
    if endpoint_status == "PIVOT_CONFIRMED":
        status_sentence = f"Окончание {last_label} подтверждено причинным pivot, но старший контекст остаётся {str(node.get('context_status', 'UNRESOLVED')).lower()}."
    else:
        status_sentence = f"Участок {last_label} ещё формируется; его структурное окончание не подтверждено."

    targets = [target for target in node.get("targets", []) if target.get("status") == "ACTIVE"]
    if targets:
        target = targets[0]
        target_sentence = f"Ближайшая расчётная зона этой модели: {target.get('price_low'):,.2f}–{target.get('price_high'):,.2f}; дата достижения не рассчитывается."
    else:
        target_sentence = "Для текущей стадии допустимая активная ценовая зона не сформирована."

    invalidation = node.get("invalidation") or {}
    if invalidation.get("level") is not None:
        invalidation_sentence = (
            f"Граница отмены — {float(invalidation['level']):,.2f} по {invalidation.get('basis', 'High/Low')}; "
            f"она относится именно к модели «{pattern}»."
        )
    else:
        invalidation_sentence = "Активная числовая граница отмены для этого узла пока не определена."

    prefix = (
        f"{scenario.get('selection_label', 'Выбранный счёт')} рассматривает {pattern} на степени "
        f"{node.get('relative_degree', visible_degree)}; глубина структурной проверки — {node.get('verified_depth', 0)}."
    )
    return " ".join([prefix, status_sentence, target_sentence, invalidation_sentence])
