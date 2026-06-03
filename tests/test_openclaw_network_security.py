import socket

from agents.openclaw.security.network import contains_internal_url, validate_url_target


def test_validate_url_target_allows_unresolved_public_hostname(monkeypatch) -> None:
    def fake_getaddrinfo(host, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    ok, reason = validate_url_target("http://repo.jfrog.org/artifactory")

    assert ok is True
    assert reason == ""


def test_contains_internal_url_blocks_private_ip_literal() -> None:
    assert contains_internal_url("curl http://127.0.0.1:8080/health") is True


def test_contains_internal_url_allows_unresolved_public_hostname(monkeypatch) -> None:
    def fake_getaddrinfo(host, *args, **kwargs):
        raise socket.gaierror(-2, "Name or service not known")

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)

    assert (
        contains_internal_url(
            "python3 -c \"print('http://repo.jfrog.org/artifactory')\""
        )
        is False
    )
