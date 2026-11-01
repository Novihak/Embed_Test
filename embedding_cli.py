#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
embedding_cli.py — универсальный CLI для получения текстовых эмбеддингов.

Поддерживаемые модели (и библиотеки, через которые они загружаются):
    1. multilingual-e5-base      -> sentence-transformers
    2. multilingual-e5-large     -> sentence-transformers
    3. bge-m3                    -> FlagEmbedding
    4. BERTA                     -> transformers (fallback: sentence-transformers)
    5. Qwen3-Embedding-0.6B      -> sentence-transformers

Зависимости (устанавливайте только те, что нужны для выбранных моделей):
    pip install sentence-transformers          # модели 1, 2, 4 (fallback), 5
    pip install FlagEmbedding                  # модель 3
    pip install transformers                   # модель 4 (основной путь)
    pip install torch                          # бэкенд для всех библиотек (обычно ставится автоматически)

Примеры запуска:
    python embedding_cli.py                                  # интерактивный режим
    python embedding_cli.py -m bge-m3 -t "Hello world"       # разовый запрос
    python embedding_cli.py -m qwen3 -f texts.txt            # пакетная обработка файла
    python embedding_cli.py --list                           # список моделей
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

__version__ = "1.0.0"

# Сколько элементов вектора печатать в консоль по умолчанию.
DEFAULT_PREVIEW_SIZE = 10


# ---------------------------------------------------------------------------
# Реестр моделей
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    """Описание одной поддерживаемой модели."""

    key: str                      # короткое имя для CLI (например, 'bge-m3')
    display_name: str             # имя для меню
    backend: str                  # используемая библиотека
    hf_repo: str                  # id репозитория на Hugging Face
    pip_package: str              # пакет для установки backend'а
    note: str = ""                # дополнительные комментарии для пользователя


MODELS: dict[str, ModelSpec] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            key="e5-base",
            display_name="multilingual-e5-base",
            backend="sentence-transformers",
            hf_repo="intfloat/multilingual-e5-base",
            pip_package="sentence-transformers",
            note="Требует префикс 'query: ' / 'passage: ' — добавляется автоматически.",
        ),
        ModelSpec(
           key="e5-large",
            display_name="multilingual-e5-large",
            backend="sentence-transformers",
            hf_repo="intfloat/multilingual-e5-large",
            pip_package="sentence-transformers",
            note="Крупная версия e5; также требует префиксы query/passage.",
        ),
        ModelSpec(
            key="bge-m3",
            display_name="bge-m3",
            backend="FlagEmbedding",
            hf_repo="BAAI/bge-m3",
            pip_package="FlagEmbedding",
            note="Загружается с use_fp16=True для экономии памяти.",
        ),
        ModelSpec(
            key="berta",
            display_name="BERTA",
            backend="transformers",
            hf_repo="sergeyzh/BERTA",
            pip_package="transformers",
            note="Робастный русскоязычный BERT (sergeyzh/BERTA); mean pooling, "
                 "fallback на sentence-transformers.",
        ),
        ModelSpec(
            key="qwen3",
            display_name="Qwen3-Embedding-0.6B",
            backend="sentence-transformers",
            hf_repo="Qwen/Qwen3-Embedding-0.6B",
            pip_package="sentence-transformers",
            note="Qwen3 поддерживает instruction-aware эмбеддинги; используются дефолтные инструкции.",
        ),
    )
}


class ModelNotFoundError(ValueError):
    """Выбранной модели нет в реестре."""


class BackendMissingError(RuntimeError):
    """Библиотека-бэкенд не установлена в окружении."""

    def __init__(self, backend: str, pip_package: str):
        super().__init__(
            f"Библиотека '{backend}' не найдена. "
            f"Установите её командой: pip install {pip_package}"
        )
        self.backend = backend
        self.pip_package = pip_package


# ---------------------------------------------------------------------------
# Абстракция бэкенда: один класс на библиотеку, без дублирования логики
# ---------------------------------------------------------------------------

def _require(module_name: str, backend: str, pip_package: str):
    """Импортирует модуль лениво; при отсутствии кидает BackendMissingError."""
    try:
        return __import__(module_name)
    except ImportError as exc:  # noqa: PERF203
        raise BackendMissingError(backend, pip_package) from exc


class BaseEmbedder(ABC):
    """Базовый класс эмбеддера: кэш модели + унифицированный интерфейс encode()."""

    #: Ключ реестра бэкенда, который реализует этот класс.
    backend_key: str = ""

    def __init__(self) -> None:
        self._models: dict[str, object] = {}  # кэш: hf_repo -> инициализированная модель

    @abstractmethod
    def _load(self, hf_repo: str) -> object:
        """Загружает модель из HF-репозитория (вызывается один раз на модель)."""

    @abstractmethod
    def _encode(self, model: object, texts: list[str]) -> list[list[float]]:
        """Вычисляет векторы для списка строк."""

    def _prepare(self, spec: ModelSpec, text: str) -> str:
        """Хук для моделей, требующих префиксов (e5, bge). По умолчанию — без изменений."""
        return text

    def get_embedding(self, spec: ModelSpec, texts: list[str]) -> list[list[float]]:
        """Публичный API: принимает тексты, возвращает список векторов."""
        model = self._models.get(spec.hf_repo)
        if model is None:
            print(f"[INFO] Загружаю модель '{spec.display_name}' ({spec.hf_repo})...")
            model = self._load(spec.hf_repo)
            self._models[spec.hf_repo] = model
        prepared = [self._prepare(spec, t) for t in texts]
        return self._encode(model, prepared)


class SentenceTransformersEmbedder(BaseEmbedder):
    """Эмбеддер на базе библиотеки sentence-transformers (e5-*, qwen3, fallback для BERTA)."""

    backend_key = "sentence-transformers"

    def _load(self, hf_repo: str):
        st = _require("sentence_transformers", self.backend_key, "sentence-transformers")
        return st.SentenceTransformer(hf_repo, trust_remote_code=True)

    def _encode(self, model, texts: list[str]) -> list[list[float]]:
        # normalize_embeddings=True делает косинусное сходство корректным.
        vectors = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return [v.tolist() for v in vectors]


class PlainSentenceTransformersEmbedder(SentenceTransformersEmbedder):
    """sentence-transformers без предобработки (Qwen3, fallback для BERTA)."""

    backend_key = "sentence-transformers"


class E5Embedder(SentenceTransformersEmbedder):
    """e5-модели требуют префиксы 'query:' / 'passage:'. Здесь — режим query."""

    backend_key = "sentence-transformers"

    def _prepare(self, spec: ModelSpec, text: str) -> str:
        if text.startswith(("query:", "passage:")):
            return text
        return f"query: {text}"


class FlagEmbeddingM3Embedder(BaseEmbedder):
    """Эмбеддер для BGE-M3 на базе FlagEmbedding (плотные векторы)."""

    backend_key = "FlagEmbedding"

    def _load(self, hf_repo: str):
        fe = _require("FlagEmbedding", self.backend_key, "FlagEmbedding")
        return fe.BGEM3Model(model_name_or_path=hf_repo, use_fp16=True)

    def _encode(self, model, texts: list[str]) -> list[list[float]]:
        output = model.encode(
            sentences=texts,
            batch_size=16,
            max_length=1024,
            return_dict=True,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"]
        return [v.tolist() for v in dense]


class TransformersBertaEmbedder(BaseEmbedder):
    """BERTA через transformers: last_hidden_state + mean pooling поверх attention mask."""

    backend_key = "transformers"

    def _load(self, hf_repo: str):
        torch = _require("torch", "torch", "torch")
        transformers = _require("transformers", self.backend_key, "transformers")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = transformers.AutoTokenizer.from_pretrained(hf_repo)
        model = transformers.AutoModel.from_pretrained(hf_repo).to(device)
        model.eval()
        return (tokenizer, model, device)

    def _encode(self, model, texts: list[str]) -> list[list[float]]:
        torch = sys.modules["torch"]
        tokenizer, net, device = model
        vectors: list[list[float]] = []
        with torch.no_grad():
            for text in texts:
                enc = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512,
                ).to(device)
                hidden = net(**enc).last_hidden_state           # (1, seq, dim)
                mask = enc["attention_mask"].unsqueeze(-1).float()
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
                vectors.append(pooled.squeeze(0).tolist())
        return vectors


# Карта: ключ бэкенда -> экземпляр эмбеддера (синглтоны дают кэш моделей).
_BACKENDS: dict[str, BaseEmbedder] = {
    "e5": E5Embedder(),                              # e5-*: ST + префиксы query/passage
    "st-plain": PlainSentenceTransformersEmbedder(), # Qwen3 и fallback BERTA: ST без префиксов
    "flagembedding": FlagEmbeddingM3Embedder(),
    "transformers": TransformersBertaEmbedder(),
}


def _resolve_embedder(spec: ModelSpec) -> BaseEmbedder:
    """Возвращает эмбдер, соответствующий модели (с fallback для BERTA)."""
    if spec.key.startswith("e5-"):
        return _BACKENDS["e5"]
    if spec.backend == "sentence-transformers":      # qwen3
        return _BACKENDS["st-plain"]
    if spec.backend == "FlagEmbedding":
        return _BACKENDS["flagembedding"]
    if spec.backend == "transformers":               # BERTA
        try:
            _require("transformers", "transformers", "transformers")
            return _BACKENDS["transformers"]
        except BackendMissingError:
            # Fallback: если transformers нет, но есть sentence-transformers — идём через него.
            _require("sentence_transformers", "sentence-transformers", "sentence-transformers")
            return _BACKENDS["st-plain"]
    raise ModelNotFoundError(f"Неизвестный бэкенд '{spec.backend}' для модели '{spec.display_name}'")


# ---------------------------------------------------------------------------
# Публичная функция, упомянутая в требованиях
# ---------------------------------------------------------------------------

def get_embedding(model_name: str, text: str | list[str]) -> list[list[float]]:
    """
    Универсальная функция получения эмбеддингов.

    Args:
        model_name: ключ или отображаемое имя модели (см. MODELS, регистронезависимо).
        text: строка или список строк.

    Returns:
        Список векторов (по одному на каждый входной текст), каждый — list[float].

    Raises:
        ModelNotFoundError: неизвестная модель.
        BackendMissingError: библиотека-бэкенд не установлена.
    """
    spec = _find_spec(model_name)
    single = isinstance(text, str)
    texts = [text] if single else list(text)
    embedder = _resolve_embedder(spec)
    return embedder.get_embedding(spec, texts)


def _find_spec(name: str) -> ModelSpec:
    """Ищет модель по ключу, display_name или номеру в меню ('1'..'5')."""
    name = name.strip()
    # По номеру в списке.
    if name.isdigit():
        specs = list(MODELS.values())
        idx = int(name) - 1
        if 0 <= idx < len(specs):
            return specs[idx]
        raise ModelNotFoundError(f"Нет модели под номером '{name}'. См. --list.")
    lowered = name.lower().replace("_", "-")
    for spec in MODELS.values():
        if lowered in (spec.key.lower(), spec.display_name.lower()):
            return spec
    raise ModelNotFoundError(
        f"Модель '{name}' не найдена. Доступны: {', '.join(m.display_name for m in MODELS.values())}"
    )


# ---------------------------------------------------------------------------
# Вывод результатов
# ---------------------------------------------------------------------------

def format_vector(vector: list[float], preview: int) -> str:
    """Форматирует вектор: первые `preview` элементов + многоточие."""
    head = ", ".join(f"{x:.6f}" for x in vector[:preview])
    if len(vector) > preview:
        head += ", ..."
    return f"[{head}]"


def report_embedding(vectors: list[list[float]], preview: int, as_json: bool) -> None:
    """Печатает размерность и значения эмбеддингов (текстом или JSON)."""
    if as_json:
        payload = [
            {"dimension": len(v), "vector_preview": v[:preview]}
            for v in vectors
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for i, vec in enumerate(vectors, start=1):
        prefix = f"[{i}] " if len(vectors) > 1 else ""
        print(f"{prefix}Размерность эмбеддинга: {len(vec)}")
        print(f"{prefix}Вектор (первые {min(preview, len(vec))} из {len(vec)}):")
        print(f"{prefix}{format_vector(vec, preview)}")


# ---------------------------------------------------------------------------
# Ввод данных: файл / stdin / аргумент
# ---------------------------------------------------------------------------

def read_texts_from_file(path: str) -> list[str]:
    """
    Читает тексты из файла.

    Форматы: .json/.jsonl — список строк или [{"text": "..."}];
    иначе — одна строка = один текст (пустые строки пропускаются).
    """
    if path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return _extract_texts(data)
    if path.endswith(".jsonl"):
        texts: list[str] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    texts.extend(_extract_texts(json.loads(line)))
        return texts
    with open(path, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def _extract_texts(data) -> list[str]:
    """Достаёт строки из разобранных JSON-данных (список строк или список объектов с 'text')."""
    if isinstance(data, str):
        return [data]
    if isinstance(data, list):
        out: list[str] = []
        for item in data:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict) and "text" in item:
                out.append(str(item["text"]))
            else:
                raise ValueError(f"Непонятный формат элемента: {item!r}")
        return out
    if isinstance(data, dict) and "texts" in data:
        return _extract_texts(data["texts"])
    raise ValueError(f"Непонятный формат JSON: ожидался список строк или объект со списком 'texts'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="embedding_cli",
        description=(
            "Получение текстовых эмбеддингов локальными моделями "
            "(multilingual-e5-base/large, bge-m3, BERTA, Qwen3-Embedding-0.6B)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-m", "--model", help="Модель: ключ, имя или номер из списка (см. --list)")
    parser.add_argument("-t", "--text", action="append", help="Текст для эмбеддинга (можно повторять)")
    parser.add_argument("-f", "--file", help="Файл со списком текстов (txt/json/jsonl) — пакетный режим")
    parser.add_argument("--list", action="store_true", help="Показать список доступных моделей и выйти")
    parser.add_argument(
        "-n", "--preview", type=int, default=DEFAULT_PREVIEW_SIZE,
        help="Сколько элементов вектора выводить в консоль",
    )
    parser.add_argument("--json", action="store_true", help="Вывести результат в формате JSON")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def print_model_list() -> None:
    """Печатает нумерованный список моделей с бэкендами и подсказками."""
    print("Доступные модели:")
    for num, spec in enumerate(MODELS.values(), start=1):
        print(f"  {num}. {spec.display_name:<24} [бэкенд: {spec.backend}]")
        if spec.note:
            print(f"     -> {spec.note}")
    print("\nУстановка недостающих библиотек: pip install sentence-transformers FlagEmbedding transformers")


def interactive_select() -> ModelSpec:
    """Интерактивный выбор модели по номеру или имени (с повтором при ошибке)."""
    print_model_list()
    while True:
        choice = input("\nВыберите модель (номер или имя): ").strip()
        try:
            return _find_spec(choice)
        except ModelNotFoundError as exc:
            print(f"[ОШИБКА] {exc}")


def run(spec: ModelSpec, texts: list[str], preview: int, as_json: bool) -> int:
    """Вычисляет и печатает эмбеддинги; возвращает код возврата."""
    if not texts:
        print("[ОШИБКА] Не передано ни одного текста (-t/--text или -f/--file).", file=sys.stderr)
        return 2
    try:
        vectors = get_embedding(spec.key, texts if len(texts) > 1 else texts[0])
    except BackendMissingError as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        return 3
    except ModelNotFoundError as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        return 4
    except Exception as exc:  # загрузка модели, OOM, сеть и т.п.
        print(
            f"[ОШИБКА] Не удалось получить эмбеддинг моделью '{spec.display_name}': {exc}\n"
            f"Проверьте, что модель доступна офлайн/онлайн и достаточно памяти.\n"
            f"При необходимости установите зависимости: pip install {spec.pip_package}",
            file=sys.stderr,
        )
        return 5
    report_embedding(vectors, preview, as_json)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """Точка входа: разбор аргументов, интерактивный режим, выполнение."""
    args = build_parser().parse_args(argv)

    if args.list:
        print_model_list()
        return 0

    # --- Выбор модели: из аргумента или интерактивно ---------------------
    try:
        spec = _find_spec(args.model) if args.model else interactive_select()
    except ModelNotFoundError as exc:
        print(f"[ОШИБКА] {exc}", file=sys.stderr)
        print_model_list()
        return 4

    # --- Сбор текстов: файл > аргументы > интерактивный ввод -------------
    if args.file:
        try:
            texts = read_texts_from_file(args.file)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[ОШИБКА] Не удалось прочитать файл '{args.file}': {exc}", file=sys.stderr)
            return 2
    elif args.text:
        texts = args.text
    else:
        print("Введите текст (пустая строка — завершение ввода):")
        texts = []
        while True:
            try:
                line = input("> ").strip()
            except EOFError:
                break
            if not line:
                break
            texts.append(line)

    return run(spec, texts, args.preview, args.json)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем.")
        sys.exit(130)
