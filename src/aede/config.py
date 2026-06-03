"""Configuration settings for AEDE."""

from pathlib import Path
from pydantic import BaseModel, Field
import os


def load_env():
    """Load .env file if it exists."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip())


# Load .env on module import
load_env()


class ModelConfig(BaseModel):
    """Model configuration - Groq for pipeline, Gemini for final reasoner."""

    # Gemini API key (for final reasoner)
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))

    # Groq API key (for extractor, analyzer, compressor)
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))

    # Pipeline model via Groq (extractor, analyzer, compressor)
    pipeline_model: str = "llama-3.3-70b-versatile"  

    # Gemini model for final reasoner
    gemini_reasoner_model: str = "gemini-2.5-flash"


class RetrievalConfig(BaseModel):
    """Retrieval configuration."""

    # ChromaDB settings
    persist_directory: Path = Path("./data/chroma_db")
    collection_name: str = "aede_documents"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 1024

    # Retrieval settings
    initial_k: int = 4
    max_k: int = 16
    binary_growth: bool = True  # k=4→8→16, not k+=4


class PipelineConfig(BaseModel):
    """Pipeline thresholds and settings."""

    # Coverage thresholds
    coverage_target: float = 0.8  # Target coverage before answering
    redundancy_threshold: float = 0.4  # Redundancy above which we compress
    confidence_threshold: float = 0.5  # Confidence below which we retrieve more

    # Compression settings
    compression_target_ratio: float = 10.0  # Target 10x reduction
    max_facts_before_compress: int = 100

    # Final reasoner settings
    target_token_input: int = 800  # Target tokens for reasoner input
    max_token_input: int = 4000  # Hard limit for reasoner input


class Settings(BaseModel):
    """All AEDE settings."""

    models: ModelConfig = Field(default_factory=ModelConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        """Load settings from YAML file."""
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        settings = cls()
        if gk := os.getenv("GEMINI_API_KEY"):
            settings.models.gemini_api_key = gk
        return settings


# Global settings instance
settings = Settings.from_env()