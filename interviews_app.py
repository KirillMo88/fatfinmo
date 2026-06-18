from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from interviews_core import (
    analyze_topic_from_summaries,
    connect_db,
    init_db,
    list_ready,
)


DB_PATH = Path(
    os.getenv("INTERVIEWS_DB_PATH", "persistent/interviews/interviews.db")
)

st.set_page_config(page_title="FatFinMo Interviews", layout="wide")
st.title("MacroVoices Interviews")
st.caption("Подробные инвестиционные саммари на русском языке")

connection = connect_db(DB_PATH)
init_db(connection)
items = list_ready(connection)
connection.close()

with st.container(border=True):
    st.subheader("Анализ мнений экспертов по теме")
    st.caption(
        "OpenAI анализирует все готовые саммари в архиве и возвращает сводку "
        "с указанием эксперта, даты и интервью."
    )
    with st.form("topic_analysis_form"):
        topic = st.text_input(
            "Тема",
            placeholder="Например: золото, нефть, Китай, ставки ФРС",
        )
        submitted = st.form_submit_button("Проанализировать")
    if submitted:
        if not topic.strip():
            st.warning("Введите тему для анализа.")
        elif not os.getenv("OPENAI_API_KEY"):
            st.error("OPENAI_API_KEY не настроен для приложения interviews.")
        else:
            with st.spinner("OpenAI анализирует архив саммари..."):
                try:
                    st.session_state["topic_analysis"] = (
                        topic,
                        analyze_topic_from_summaries(items, topic),
                    )
                except Exception as exc:
                    st.error(f"Не удалось выполнить анализ: {exc}")
    if "topic_analysis" in st.session_state:
        analyzed_topic, analysis = st.session_state["topic_analysis"]
        st.markdown(f"### Тема: {analyzed_topic}")
        st.markdown(analysis)

all_speakers = sorted(
    {speaker for item in items for speaker in item["speakers"]},
    key=str.casefold,
)
selected = st.multiselect(
    "Спикеры",
    all_speakers,
    format_func=lambda value: f"@{value}",
    placeholder="Выберите одного или нескольких спикеров",
)
query = st.text_input(
    "Поиск по архиву",
    placeholder="Название, спикер или текст саммари",
)

left, right = st.columns([1, 5])
with left:
    if st.button("Сбросить фильтры", use_container_width=True):
        st.session_state.clear()
        st.rerun()
with right:
    st.caption(
        "Теги спикеров: "
        + ("  ".join(f"`@{speaker}`" for speaker in all_speakers) or "пока нет")
    )

needle = query.casefold().strip()
filtered = []
for item in items:
    if selected and not set(selected).issubset(item["speakers"]):
        continue
    haystack = " ".join(
        [item["title"], item["summary"], *item["speakers"]]
    ).casefold()
    if needle and needle not in haystack:
        continue
    filtered.append(item)

st.divider()
st.caption(f"Найдено интервью: {len(filtered)}")
for item in filtered:
    date_text = item["published_at"][:10]
    tags = " ".join(f"@{speaker}" for speaker in item["speakers"])
    with st.expander(f"{date_text} · {item['title']}", expanded=False):
        st.caption(tags)
        st.markdown(item["summary"])
        st.link_button("Открыть оригинальный транскрипт", item["url"])

if not items:
    st.info("Архив создается. Первые саммари появятся после завершения обработки.")
