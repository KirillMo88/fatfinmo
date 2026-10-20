from __future__ import annotations

import os
from datetime import date
from typing import Any, Callable, Mapping

import streamlit as st

from weekly_market_intelligence import DEFAULT_DB_PATH, IntelligenceRepository


LIFECYCLE_LABELS = {
    "NEW": "Новая",
    "DEVELOPING": "Развивается",
    "ESTABLISHED": "Устоявшаяся",
    "WEAKENING": "Ослабевает",
    "RESOLVED": "Завершена",
    "INVALIDATED": "Опровергнута",
}


def _event_label(event: Mapping[str, Any]) -> str:
    groups = ", ".join(event.get("groupIds", []))
    return f"{event.get('type', 'EVENT')} — {groups}" if groups else str(event.get("type", "EVENT"))


def _render_thesis(thesis: Mapping[str, Any], events: Mapping[str, Mapping[str, Any]]) -> None:
    lifecycle = thesis.get("lifecycle", thesis.get("proposedLifecycle", "NEW"))
    assessment = thesis.get("evidenceAssessment", "PRELIMINARY")
    st.markdown(f"### {thesis.get('headline', 'Гипотеза')}")
    st.caption(
        f"{LIFECYCLE_LABELS.get(lifecycle, lifecycle)} · {assessment} · "
        + ", ".join(thesis.get("groupIds", []))
    )
    st.write(thesis.get("interpretation", ""))
    st.markdown(f"**Почему это важно:** {thesis.get('whyItMatters', '')}")
    alternative = thesis.get("alternativeInterpretation")
    if alternative:
        st.markdown(f"**Альтернативная интерпретация:** {alternative}")
    with st.expander("Доказательства и условия", expanded=False):
        support = thesis.get("supportingEventIds", [])
        contradicting = thesis.get("contradictingEventIds", [])
        st.markdown("**Поддерживающие события**")
        for event_id in support:
            event = events.get(event_id)
            st.markdown(f"- {_event_label(event) if event else event_id}")
        st.markdown("**Противоречащие события / ограничения**")
        if contradicting:
            for event_id in contradicting:
                event = events.get(event_id)
                st.markdown(f"- {_event_label(event) if event else event_id}")
        else:
            st.markdown("- Отдельное противоречащее событие не выбрано; см. альтернативную интерпретацию и ограничения отчёта.")
        st.markdown("**Подтверждение**")
        for condition_id in thesis.get("confirmationConditionIds", []):
            st.markdown(f"- `{condition_id}`")
        st.markdown("**Инвалидация**")
        for condition_id in thesis.get("invalidationConditionIds", []):
            st.markdown(f"- `{condition_id}`")


def _changed_this_week(report: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [
        item
        for item in (report.get("lifecycle_changes") or [])
        if item.get("from") != item.get("to")
    ]


def render_weekly_market_intelligence(manual_run: Callable[[], Any] | None = None) -> None:
    st.markdown("## Weekly Market Intelligence")
    st.caption("Еженедельный LLM-синтез только по агрегированным Group 2. Открытие этого блока не выполняет запросы к модели.")
    try:
        repository = IntelligenceRepository(os.getenv("WMI_DB_PATH", str(DEFAULT_DB_PATH)))
        reports = repository.list_published()
        latest_status = repository.latest_run_status()
    except Exception as exc:
        st.warning(f"Хранилище Weekly Market Intelligence пока недоступно: {exc}")
        return

    action_col, note_col = st.columns([1.25, 3.75], vertical_alignment="center")
    with action_col:
        manual_clicked = st.button(
            "Запустить LLM вручную",
            key="weekly_market_intelligence_manual_run",
            type="primary",
            use_container_width=True,
            disabled=manual_run is None,
        )
    with note_col:
        st.caption(
            "Использует последнюю полностью завершённую пятницу. Уже опубликованная неделя повторно не расходует токены."
        )
    flash = st.session_state.pop("weekly_market_intelligence_manual_flash", None)
    if flash:
        getattr(st, flash.get("kind", "info"))(flash.get("message", ""))
    if manual_clicked and manual_run is not None:
        try:
            with st.spinner("Проверяю данные и формирую Weekly Market Intelligence..."):
                result = manual_run()
            if result.status == "PUBLISHED":
                used_llm = int(result.details.get("events", 0)) > 0
                st.session_state["weekly_market_intelligence_manual_flash"] = {
                    "kind": "success",
                    "message": (
                        f"Отчёт за {result.week_end} опубликован."
                        + (" LLM-анализ выполнен." if used_llm else " Значимых событий не было, поэтому LLM-вызов не потребовался.")
                    ),
                }
                st.rerun()
            elif result.status == "SKIPPED_ALREADY_PUBLISHED":
                st.info(f"Отчёт за {result.week_end} уже опубликован — повторный LLM-вызов не выполнен.")
            elif result.status == "SKIPPED_RUN_IN_PROGRESS":
                st.info(f"Формирование отчёта за {result.week_end} уже выполняется в другом процессе.")
            elif result.status == "DATA_NOT_READY":
                complete = result.details.get("completeGroups", 0)
                total = result.details.get("groupCount", 0)
                st.warning(f"Данные за {result.week_end} ещё не готовы: полное покрытие у {complete} из {total} групп.")
            else:
                st.error(f"Ручной запуск завершён со статусом {result.status}.")
        except Exception as exc:
            st.error(f"Ручной запуск не выполнен: {exc}")
    if not reports:
        if latest_status and latest_status.get("status") != "PUBLISHED":
            st.warning(
                f"Новый отчёт пока не опубликован: {latest_status.get('status')} "
                f"(неделя {latest_status.get('week_end')}). Предыдущего валидного отчёта ещё нет."
            )
        else:
            st.info("Опубликованных недельных отчётов пока нет. Первый отчёт появится после субботней проверки готовности данных.")
        return

    if latest_status and latest_status.get("status") != "PUBLISHED" and latest_status.get("week_end") >= reports[0]["week_end"]:
        st.warning(
            f"Показан последний валидный отчёт. Обновление за {latest_status.get('week_end')} имеет статус "
            f"{latest_status.get('status')}; данные или генерация ещё не готовы."
        )

    labels = {item["week_end"]: item for item in reports}
    selected_week = st.selectbox(
        "Отчётная неделя",
        options=list(labels),
        index=0,
        key="weekly_market_intelligence_week",
    )
    report = labels[selected_week]
    output = report.get("output") or {}
    events_list = repository.load_events(date.fromisoformat(selected_week))
    events = {item["eventId"]: item for item in events_list}
    snapshots = repository.load_snapshots(date.fromisoformat(selected_week))
    condition_registry = {
        item["conditionId"]: item
        for event in events_list
        for item in event.get("conditionDefinitions", [])
    }

    st.caption(
        f"Неделя: {selected_week} · Создан: {report.get('generation_timestamp', '—')} · "
        f"Опубликован: {report.get('publication_timestamp', '—')}"
    )

    executive = output.get("executive", {})
    st.markdown(f"### {executive.get('headline', 'Общая интерпретация')}")
    st.write(executive.get("interpretation", ""))

    st.markdown("### Основные рыночные гипотезы")
    theses = output.get("theses", [])
    if not theses:
        st.info("Новых гипотез достаточной материальности на этой неделе не опубликовано.")
    for thesis in theses[:3]:
        with st.container(border=True):
            _render_thesis(thesis, events)

    st.markdown("### Что изменилось за неделю")
    changes = _changed_this_week(report)
    assessment = output.get("weeklyAssessment", {})
    st.write(assessment.get("interpretation", ""))
    if changes:
        for change in changes:
            before = LIFECYCLE_LABELS.get(change.get("from"), change.get("from") or "—")
            after = LIFECYCLE_LABELS.get(change.get("to"), change.get("to") or "—")
            st.markdown(f"- `{change.get('themeKey')}`: {before} → {after}")
    elif assessment.get("materialChangeDetected"):
        st.caption("Материальное изменение отражено в новых событиях, без смены жизненного цикла существующих гипотез.")
    else:
        st.caption("Материальных изменений жизненного цикла гипотез не выявлено.")

    st.markdown("### Аналитика по Group 2")
    all_groups = sorted(item["groupId"] for item in snapshots)
    selected_group = st.selectbox("Group 2", options=all_groups, key="weekly_market_intelligence_group") if all_groups else None
    related = [thesis for thesis in theses if selected_group in thesis.get("groupIds", [])] if selected_group else []
    if selected_group and related:
        for thesis in related:
            st.markdown(f"- **{thesis.get('headline')}** — {thesis.get('interpretation')}")
    elif selected_group:
        st.info("Для этой группы в последнем недельном отчёте не выявлено нового существенного аналитического развития.")
    else:
        st.info("В отчёте нет групп с опубликованными значимыми событиями.")

    st.markdown("### Что отслеживать дальше")
    monitoring = []
    for thesis in theses:
        for condition_id in thesis.get("confirmationConditionIds", []):
            monitoring.append(("Подтверждение", condition_id, thesis))
        for condition_id in thesis.get("invalidationConditionIds", []):
            monitoring.append(("Инвалидация", condition_id, thesis))
    if monitoring:
        for kind, condition_id, thesis in monitoring:
            description = condition_registry.get(condition_id, {}).get("description", condition_id)
            st.markdown(
                f"- **{kind}:** {description} · {thesis.get('headline')} · "
                + ", ".join(thesis.get("groupIds", []))
            )
    else:
        st.caption("Активных условий подтверждения или инвалидации в выбранном отчёте нет.")

    limitations = output.get("limitations", [])
    if limitations:
        with st.expander("Ограничения и диагностика", expanded=False):
            for limitation in limitations:
                st.markdown(f"- {limitation}")
            st.caption(
                f"Модель: {report.get('model_identifier')} · события: {len(report.get('selected_event_ids') or [])} · "
                f"tokens: {report.get('total_tokens') or 0} · estimated cost: {report.get('estimated_cost') or 0:.6f}"
            )

    if len(reports) > 1:
        previous = next((item for item in reports if item["week_end"] < selected_week), None)
        if previous:
            with st.expander("Сравнение с предыдущим опубликованным отчётом", expanded=False):
                previous_output = previous.get("output") or {}
                st.caption(f"Предыдущая неделя: {previous['week_end']}")
                st.write(previous_output.get("executive", {}).get("interpretation", ""))
