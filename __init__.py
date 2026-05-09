# uplan/extraction/__init__.py
try:
    from .extractor import DocumentExtractor, MockVLMBackend
    from .models import (
        DocumentExtractionResult,
        Confidence,
        ConfidenceWrapper,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )
except ImportError:
    from extractor import DocumentExtractor, MockVLMBackend
    from models import (
        DocumentExtractionResult,
        Confidence,
        ConfidenceWrapper,
        PageExtractionResult,
        PageType,
        SourceQuality,
    )

__all__ = [
    "DocumentExtractor",
    "MockVLMBackend",
    "DocumentExtractionResult",
    "Confidence",
    "ConfidenceWrapper",
    "PageExtractionResult",
    "PageType",
    "SourceQuality",
]
