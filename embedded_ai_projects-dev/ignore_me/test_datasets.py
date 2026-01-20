#!/usr/bin/env python3
import json
import os
import socket
import unittest
import urllib.error
import urllib.request


def load_projects():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(root, "ignore", "datasets.json")
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data.get("projects", {})


def network_available(timeout=2):
    try:
        with socket.create_connection(("1.1.1.1", 443), timeout=timeout):
            return True
    except OSError:
        return False


def can_open_url(url, timeout=10):
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; dataset-link-check/1.0)"
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        status = getattr(resp, "status", 200)
        if status >= 400:
            raise urllib.error.HTTPError(
                url, status, "HTTP error", resp.headers, None
            )
        resp.read(64)


class DatasetLinkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not network_available():
            raise unittest.SkipTest("Network unavailable; skipping dataset URL checks.")

    def test_dataset_links_reachable(self):
        projects = load_projects()
        failures = []
        for key in sorted(projects.keys(), key=lambda x: int(x)):
            project = projects[key]
            url = project.get("dataset_link")
            if not url:
                failures.append(f"{key}: missing dataset_link")
                continue
            try:
                can_open_url(url)
            except Exception as exc:
                failures.append(f"{key}: {url} -> {exc}")
        if failures:
            message = "\n".join(failures)
            self.fail(f"Dataset link check failed:\n{message}")


if __name__ == "__main__":
    unittest.main()
