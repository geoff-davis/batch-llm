"""ProcessorConfig proactive-rate validation regressions (issue #147)."""

import logging

import pytest

from async_batch_llm import ProcessorConfig


@pytest.mark.parametrize("rpm", [10.0, 59.9, 1.0, 0.25])
def test_low_positive_rpm_with_multiple_workers_is_valid_and_quiet(
    rpm: float, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="async_batch_llm.core.config"):
        config = ProcessorConfig(max_workers=2, max_requests_per_minute=rpm)

    assert config.max_requests_per_minute == rpm
    assert not caplog.records
