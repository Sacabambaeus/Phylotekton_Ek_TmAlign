#!/usr/bin/env python3
"""Download FASTA sequences from NCBI nuccore for a predefined query."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterable, List
from http import client
from urllib import error, parse, request

EUTILS_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
DEFAULT_QUERY = (
    "Animals[filter] OR Plants[filter] OR Fungi[filter] OR Protists[filter] "
    "NOT draft genome[All Fields] NOT genome assembly[All Fields] "
    "NOT operon gene[All Fields] NOT partial[All Fields] NOT chloroplast[All Fields] "
    "AND (complete[All Fields] OR (12S[All Fields] AND ribosomal[All Fields]))"
)


class NcbiFastaDownloader:
    """Utility class that wraps the NCBI E-utilities workflow."""

    def __init__(
        self,
        query: str,
        output_path: str,
        chunk_size: int = 200,
        max_records: int | None = None,
        email: str | None = None,
        api_key: str | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be positive when provided")

        self.query = query
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.max_records = max_records
        self.email = email
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.tool_name = "ncbi-fasta-downloader"
        self.min_delay = 0.11 if self.api_key else 0.34  # follow NCBI rate guidance
        self.retries = 5
        self.retry_backoff = 1.5
        self.timeout = 60
        self._last_request_ts = 0.0
        ua_email = self.email or "anonymous@example.com"
        self.user_agent = f"{self.tool_name}/1.0 ({ua_email})"

    def download(self) -> None:
        total_records = None
        fetched = 0
        retstart = 0

        with open(self.output_path, "w", encoding="utf-8") as handle:
            while True:
                if self.max_records is not None and fetched >= self.max_records:
                    break
                retmax = self.chunk_size
                if self.max_records is not None:
                    retmax = min(retmax, self.max_records - fetched)
                data = self._esearch_with_retry(retstart=retstart, retmax=retmax)
                total_records = total_records or int(
                    data["esearchresult"].get("count", 0)
                )
                idlist: List[str] = data["esearchresult"].get("idlist", [])
                if not idlist:
                    break

                fasta_text = self._efetch(idlist)
                if fasta_text and not fasta_text.endswith("\n"):
                    fasta_text += "\n"
                handle.write(fasta_text)
                fetched += len(idlist)
                retstart = fetched
                progress = f"{fetched} sequences"
                if total_records:
                    progress = f"{fetched}/{total_records} sequences"
                print(f"Downloaded {progress}...", file=sys.stderr)

        if fetched == 0:
            raise RuntimeError("Query returned no records; nothing was written.")
        print(
            f"Saved {fetched} sequences to {self.output_path}",
            file=sys.stderr,
        )

    def _esearch_with_retry(self, retstart: int, retmax: int) -> dict:
        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                return self._esearch(retstart=retstart, retmax=retmax)
            except RuntimeError as exc:
                last_exc = exc
                if attempt == self.retries - 1:
                    raise
                delay = self.retry_backoff ** attempt
                print(
                    f"Esearch failed ({exc}); retrying in {delay:.1f}s",
                    file=sys.stderr,
                )
                time.sleep(delay)
        raise RuntimeError(f"Esearch repeatedly failed: {last_exc}")

    def _esearch(self, retstart: int, retmax: int) -> dict:
        params = {
            "db": "nuccore",
            "term": self.query,
            "retmode": "json",
            "retmax": retmax,
            "retstart": retstart,
            "tool": self.tool_name,
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        payload = self._request("esearch.fcgi", params)
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            snippet = payload.strip().splitlines()
            preview = " ".join(snippet[:3])[:400]
            raise RuntimeError(
                "NCBI esearch payload was not valid JSON; "
                f"response preview: {preview!r}"
            ) from exc
        error_msg = data.get("esearchresult", {}).get("ERROR")
        if error_msg:
            raise RuntimeError(f"NCBI esearch returned error: {error_msg}")
        return data

    def _efetch(self, ids: Iterable[str]) -> str:
        params = {
            "db": "nuccore",
            "id": ",".join(ids),
            "rettype": "fasta",
            "retmode": "text",
            "tool": self.tool_name,
        }
        if self.email:
            params["email"] = self.email
        if self.api_key:
            params["api_key"] = self.api_key
        return self._request("efetch.fcgi", params)

    def _request(self, endpoint: str, params: dict) -> str:
        # Enforce polite spacing between requests.
        sleep_for = self.min_delay - (time.time() - self._last_request_ts)
        if sleep_for > 0:
            time.sleep(sleep_for)

        url = f"{EUTILS_BASE}{endpoint}?{parse.urlencode(params)}"
        req = request.Request(url, headers={"User-Agent": self.user_agent})

        last_exc: Exception | None = None
        for attempt in range(self.retries):
            try:
                with request.urlopen(req, timeout=self.timeout) as response:
                    payload = response.read().decode("utf-8")
                self._last_request_ts = time.time()
                return payload
            except error.HTTPError as exc:  # pragma: no cover - network error path
                last_exc = exc
                should_retry = exc.code in {429, 500, 502, 503, 504}
            except error.URLError as exc:  # pragma: no cover - network error path
                last_exc = exc
                should_retry = True
            except client.IncompleteRead as exc:  # pragma: no cover - network error path
                last_exc = exc
                should_retry = True
            else:
                should_retry = False

            if not should_retry or attempt == self.retries - 1:
                break
            backoff = self.retry_backoff ** attempt
            time.sleep(backoff)

        if last_exc:
            raise RuntimeError(f"NCBI request failed: {last_exc}")
        raise RuntimeError("NCBI request failed for an unknown reason")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="nuccore_sequences.fasta",
        help="Path to the FASTA file to create (default: %(default)s)",
    )
    parser.add_argument(
        "--query",
        default=DEFAULT_QUERY,
        help="Entrez query string to run",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help=(
            "How many IDs to fetch per request (default: %(default)s). "
            "Lower this if you encounter HTTP 414 errors."
        ),
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Limit the number of sequences downloaded (useful for testing)",
    )
    parser.add_argument(
        "--email",
        default=None,
        help="Contact email passed to NCBI (recommended)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="NCBI API key to raise the rate limit (or set NCBI_API_KEY env)",
    )
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    downloader = NcbiFastaDownloader(
        query=args.query,
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_records=args.max_records,
        email=args.email,
        api_key=args.api_key,
    )
    downloader.download()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
