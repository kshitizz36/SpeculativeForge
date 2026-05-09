from __future__ import annotations

from typing import Any, Optional

import requests

from speculate_forge.models import (
    SpeculateAction,
    SpeculateConfig,
    SpeculateResetResponse,
    SpeculateStateResponse,
    SpeculateStepResponse,
)


class SpeculateForgeEnv:
    """Thin HTTP client for the SpeculateForge environment."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json: Optional[dict[str, Any]] = None,
    ) -> Any:
        response = requests.request(
            method=method,
            url=f"{self.base_url}{path}",
            params=params,
            json=json,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def info(self) -> dict[str, Any]:
        return self._request("GET", "/info")

    def schema(self) -> dict[str, Any]:
        return self._request("GET", "/schema")

    def metadata(self) -> dict[str, Any]:
        return self._request("GET", "/metadata")

    def state(self) -> SpeculateStateResponse:
        return SpeculateStateResponse.model_validate(self._request("GET", "/state"))

    def reset(self, task_level: int = 1) -> SpeculateResetResponse:
        return SpeculateResetResponse.model_validate(
            self._request("POST", "/reset", params={"task_level": task_level})
        )

    def step(
        self, action: SpeculateAction | dict[str, Any], task_level: int = 1
    ) -> SpeculateStepResponse:
        payload = action.model_dump() if isinstance(action, SpeculateAction) else action
        return SpeculateStepResponse.model_validate(
            self._request("POST", "/step", params={"task_level": task_level}, json=payload)
        )

    def manual_step(
        self, config: SpeculateConfig | dict[str, Any], task_level: int = 1
    ) -> dict[str, Any]:
        payload = config.model_dump() if isinstance(config, SpeculateConfig) else config
        return self._request(
            "POST",
            "/manual_step",
            params={"task_level": task_level},
            json=payload,
        )
