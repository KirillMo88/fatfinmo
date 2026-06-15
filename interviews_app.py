from __future__ import annotations

import os
from pathlib import Path

import streamlit as st

from interviews_core import connect_db, init_db, list_ready


DB_PATH = Path(
    os.getenv("INTERVIEWS_DB_PATH", "persistent/interviews/interviews.db")
)

st.set_page_config(page_title="FatFinMo Interviews", page_icon="🎙️", layout="wide")
st.title("MacroVoices Interviews")
st.caption("Подробные инвестиционные саммари на русском языке")

connection = connect_db(DB_PATH)
init_db(connection)
items = list_ready(connection)
connection.close()

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
query = st.text_input("Поиск", placeholder="Название, спикер или текст саммари")

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
