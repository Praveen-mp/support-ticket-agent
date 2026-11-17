"""Settings from environment variables (and a local .env file). Validated at start-up."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent

Provider = Literal["groq", "anthropic"]


@dataclass(frozen=True)
class ProviderDefaults:
    model: str
    price_input_per_mtok: float
    price_output_per_mtok: float
    cache_read_factor: float
    cache_write_factor: float


# Published list prices (USD per million tokens), used for the cost report.
# Groq's free tier is not billed; its price shows what the same run would cost when paid.
PROVIDER_DEFAULTS: dict[Provider, ProviderDefaults] = {
    "groq": ProviderDefaults("openai/gpt-oss-120b", 0.15, 0.60, 0.5, 1.0),
    "anthropic": ProviderDefaults("claude-haiku-5-5", 0.10, 0.50, 0.1, 1.25),
}


class Settings(BaseSettings):
    # env_ignore_empty: a blank line like `LLM_MODEL=` means "use the default".
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", extra="ignore", env_ignore_empty=True
    )

    helpdesk_base_url: str = "https://nakywgkztvzkihdjnwbc.supabase.co/functions/v1/helpdesk"
    # Optional here so `eval` can run from the saved examples without helpdesk access.
    # Commands that call the helpdesk check them before starting.
    helpdesk_api_key: SecretStr | None = None
    candidate_email: str = ""
    candidate_name: str = ""

    llm_provider: Provider = "groq"
    llm_model: str = ""  # empty: the provider's default model
    llm_effort: Literal["low", "medium", "high"] = "low"
    # Empty: the provider's published list price.
    price_input_per_mtok: float | None = Field(default=None, ge=0)
    price_output_per_mtok: float | None = Field(default=None, ge=0)
    groq_api_key: SecretStr | None = None
    groq_base_url: str = "https://api.groq.com/openai/v1"
    model_requests_per_minute: float = Field(default=25, gt=0)
    anthropic_api_key: SecretStr | None = None

    confidence_threshold: float = Field(default=0.75, ge=0, le=1)
    requests_per_second: float = Field(default=4.0, gt=0, le=5)
    max_run_cost_usd: float = Field(default=0.50, gt=0)
    max_tickets_per_run: int = Field(default=200, gt=0)

    faq_path: Path = PROJECT_ROOT / "data" / "faq.md"
    log_dir: Path = PROJECT_ROOT / "logs"
    state_path: Path = PROJECT_ROOT / "state" / "decisions.json"
    eval_dir: Path = PROJECT_ROOT / "eval"

    @property
    def model_name(self) -> str:
        return self.llm_model or PROVIDER_DEFAULTS[self.llm_provider].model

    @property
    def provider_defaults(self) -> ProviderDefaults:
        return PROVIDER_DEFAULTS[self.llm_provider]
