# Python Project Coding Standards

> This document defines the coding conventions, style guidelines, and architectural rules for this Python project.
> All contributors and AI coding agents must follow these rules consistently.

---

## 1. General Philosophy

- **Clarity over cleverness** — code is read far more often than it is written.
- **Explicit over implicit** — never rely on hidden behavior or magic defaults.
- **Single Responsibility** — every class and function does one thing and does it well.
- **Fail loudly** — raise meaningful exceptions rather than silently swallowing errors.

---

## 2. Language & Tooling

| Tool | Version / Standard |
|------|--------------------|
| Python | ≥ 3.11 |
| Package manager | `uv` (preferred) or `pip` |
| Linter | `ruff` |
| Formatter | `ruff format` |
| Type checker | `mypy --strict` |
| Test runner | `pytest` |

---

## 4. Type Hints

**All** functions and methods must be fully type-annotated — parameters, return types, and class attributes.

```python
# ✅ Correct
def extract_text(image_path: Path, language: str = "vie") -> str:
    ...

# ❌ Wrong
def extract_text(image_path, language="vie"):
    ...
```

### Rules

- Use `from __future__ import annotations` at the top of every file to enable PEP 563 deferred evaluation.
- Prefer built-in generics (`list[str]`, `dict[str, int]`) over `typing.List`, `typing.Dict` (Python ≥ 3.9).
- Use `X | None` instead of `Optional[X]` (Python ≥ 3.10).
- Use `X | Y` instead of `Union[X, Y]`.
- Never use `Any` unless truly unavoidable; add a `# type: ignore[assignment]` comment with a reason if you must.

```python
from __future__ import annotations

from pathlib import Path


def load_model(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    ...


def parse_result(raw: dict[str, object]) -> ParsedResult | None:
    ...
```

---

## 5. Docstrings

Every **public** class, method, and function must have a docstring written **in English** using the Google style.

### Function / Method template

```python
def process_document(
    file_path: Path,
    language: str = "vie",
    *,
    dpi: int = 300,
) -> DocumentResult:
    """Process a document file and extract structured text.

    Args:
        file_path: Absolute path to the input file (PDF, DOCX, or DOC).
        language: Tesseract language code used for OCR. Defaults to "vie".
        dpi: Resolution (dots per inch) for PDF rasterisation. Defaults to 300.

    Returns:
        A DocumentResult containing the extracted text and metadata.

    Raises:
        FileNotFoundError: If ``file_path`` does not exist.
        UnsupportedFormatError: If the file extension is not supported.
    """
```

### Class template

```python
class OCRService:
    """Orchestrates OCR processing for Vietnamese administrative documents.

    Combines PDF rasterisation, DBNet-based text detection, and VietOCR
    recognition into a single pipeline.

    Attributes:
        model_path: Path to the ONNX detection model.
        device: Torch device string, e.g. ``"cpu"`` or ``"cuda:0"``.
    """

    model_path: Path
    device: str
```

### Rules

- One-line summary ends with a period.
- Leave a blank line between the summary and the Args / Returns / Raises sections.
- Private helpers (`_foo`) may have a short one-liner docstring; internal logic comments use `#`.

---

## 6. Classes

### 6.1 Definition rules

- One class per file unless the classes are tightly coupled (e.g., a small private helper).
- Use `@dataclass` or Pydantic `BaseModel` for data-holder classes — avoid writing `__init__` manually for plain data.
- Prefer composition over inheritance. Maximum inheritance depth: **2**.
- Define abstract base classes with `abc.ABC` and `@abc.abstractmethod`.

### 6.2 Method ordering inside a class

Follow this order:

1. Class variables / `ClassVar` annotations
2. `__init__` (or Pydantic field declarations)
3. `@classmethod` / `@staticmethod` factory methods
4. Public methods (alphabetical within logical groups)
5. Private methods (`_name`) (alphabetical)
6. `__dunder__` methods other than `__init__`

### 6.3 Example

```python
from __future__ import annotations

import abc
from pathlib import Path

import torch


class BaseDetector(abc.ABC):
    """Abstract interface for text detection models."""

    @abc.abstractmethod
    def detect(self, image: torch.Tensor) -> list[BoundingBox]:
        """Detect text regions in an image tensor.

        Args:
            image: A float32 tensor of shape (C, H, W) normalised to [0, 1].

        Returns:
            List of detected bounding boxes sorted top-to-bottom, left-to-right.
        """


class DBNetDetector(BaseDetector):
    """DBNet text detector backed by an ONNX runtime session.

    Attributes:
        model_path: Path to the exported ``*.onnx`` model file.
        threshold: Binarisation threshold for the probability map.
    """

    def __init__(self, model_path: Path, threshold: float = 0.3) -> None:
        """Initialise the detector and load the ONNX model.

        Args:
            model_path: Path to the ONNX model file.
            threshold: Binarisation threshold. Defaults to 0.3.

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model_path = model_path
        self.threshold = threshold
        self._session = self._load_session()

    @classmethod
    def from_pretrained(cls, name: str) -> DBNetDetector:
        """Load a detector from the local model registry by name.

        Args:
            name: Registry key for the model (e.g. ``"dbnet-viet-v1"``).

        Returns:
            An initialised DBNetDetector instance.
        """
        path = _resolve_model_path(name)
        return cls(model_path=path)

    def detect(self, image: torch.Tensor) -> list[BoundingBox]:
        """Detect text regions in an image tensor.

        Args:
            image: Float32 tensor of shape (C, H, W) normalised to [0, 1].

        Returns:
            Bounding boxes sorted top-to-bottom, left-to-right.
        """
        raw = self._session.run(None, {"input": image.numpy()})
        return self._postprocess(raw[0])

    def _load_session(self) -> ort.InferenceSession:
        """Load and return the ONNX InferenceSession."""
        return ort.InferenceSession(str(self.model_path))

    def _postprocess(self, prob_map: np.ndarray) -> list[BoundingBox]:
        """Convert a probability map to a list of bounding boxes.

        Args:
            prob_map: Float32 array of shape (H, W).

        Returns:
            Filtered and sorted bounding boxes.
        """
        ...
```

---

## 7. Functions & Methods

- Maximum function length: **40 lines** of logic (excluding docstring). Extract helpers if exceeded.
- Maximum cyclomatic complexity: **10**. Use early returns and guard clauses to reduce nesting.
- Keyword-only arguments for optional params that affect behaviour — use `*` separator.
- Never use mutable default arguments (`list`, `dict`). Use `None` and initialise inside the body.

```python
# ✅ Correct
def build_prompt(
    context: str,
    *,
    max_tokens: int = 512,
    language: str = "vi",
) -> str:
    ...

# ❌ Wrong — mutable default
def append_item(item: str, container: list[str] = []) -> list[str]:
    ...
```

---

## 8. Naming Conventions

| Entity | Convention | Example |
|--------|-----------|---------|
| Module / package | `snake_case` | `ocr_service.py` |
| Class | `PascalCase` | `DocumentParser` |
| Function / method | `snake_case` | `extract_metadata()` |
| Variable | `snake_case` | `page_count` |
| Constant | `UPPER_SNAKE_CASE` | `MAX_RETRIES = 3` |
| Private attr/method | `_single_leading_underscore` | `_session` |
| Type alias | `PascalCase` | `BoundingBox = tuple[int, int, int, int]` |
| Generic TypeVar | Single uppercase letter or descriptive | `T`, `ModelT` |

---

## 10. Logging

- Use loguru — never use print() in production code.
- Import the singleton directly: from loguru import logger. No per-module setup needed.
- Use f-string style in log calls (loguru is lazy by default — no performance concern).
- Log levels: DEBUG for internal state, INFO for milestones, WARNING for recoverable issues, ERROR for failures, EXCEPTION inside except blocks to auto-attach the traceback.
- Configure sinks once at application entry point (main.py or lifespan). Never configure inside library modules.

---

## 11. Configuration

- All configuration comes from environment variables via `pydantic-settings`.
- No hard-coded secrets or paths in source code.
- Provide `.env.example` with every key documented.

```python
# src/<pkg>/config.py
from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    ocr_model_path: Path
    llm_api_key: str
    s3_bucket_name: str
    log_level: str = "INFO"


settings = Settings()  # singleton imported by other modules
```

---

## 12. Testing

- Minimum coverage target: **80 %** for the `core/` and `services/` layers.
- Unit tests must not perform I/O — mock all external calls with `unittest.mock` or `pytest-mock`.
- Name test functions descriptively: `test_<unit>_<scenario>_<expected_outcome>`.

```python
def test_extract_metadata_with_valid_pdf_returns_doc_number() -> None:
    ...

def test_extract_metadata_with_missing_file_raises_file_not_found() -> None:
    ...
```

---

## 13. Imports

Order (enforced by `ruff`):

1. `__future__`
2. Standard library
3. Third-party packages
4. Local application imports

Separate each group with a blank line. Use absolute imports only.

```python
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel

from mypackage.config import settings
from mypackage.models.document import DocumentResult
```


