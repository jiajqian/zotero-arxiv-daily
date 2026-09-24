from .base import BaseRetriever, register_retriever
import arxiv
from arxiv import Result as ArxivResult
from ..protocol import Paper
from ..utils import extract_markdown_from_pdf, extract_tex_code_from_tar
from tempfile import TemporaryDirectory
import feedparser
from tqdm import tqdm
import multiprocessing
import os
import re
import calendar
from datetime import datetime, timezone
from queue import Empty
from time import sleep
from typing import Any, Callable, TypeVar
from loguru import logger
import requests

T = TypeVar("T")

DOWNLOAD_TIMEOUT = (10, 60)
PDF_EXTRACT_TIMEOUT = 180
TAR_EXTRACT_TIMEOUT = 180


def _download_file(url: str, path: str) -> None:
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as response:
        response.raise_for_status()
        with open(path, "wb") as file:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    file.write(chunk)


def _run_in_subprocess(
    result_queue: Any,
    func: Callable[..., T | None],
    args: tuple[Any, ...],
) -> None:
    try:
        result_queue.put(("ok", func(*args)))
    except Exception as exc:
        result_queue.put(("error", f"{type(exc).__name__}: {exc}"))


def _run_with_hard_timeout(
    func: Callable[..., T | None],
    args: tuple[Any, ...],
    *,
    timeout: float,
    operation: str,
    paper_title: str,
) -> T | None:
    start_methods = multiprocessing.get_all_start_methods()
    context = multiprocessing.get_context("fork" if "fork" in start_methods else start_methods[0])
    result_queue = context.Queue()
    process = context.Process(target=_run_in_subprocess, args=(result_queue, func, args))
    process.start()

    try:
        status, payload = result_queue.get(timeout=timeout)
    except Empty:
        if process.is_alive():
            process.kill()
        process.join(5)
        result_queue.close()
        result_queue.join_thread()
        logger.warning(f"{operation} timed out for {paper_title} after {timeout} seconds")
        return None

    process.join(5)
    result_queue.close()
    result_queue.join_thread()

    if status == "ok":
        return payload

    logger.warning(f"{operation} failed for {paper_title}: {payload}")
    return None


def _extract_text_from_pdf_worker(pdf_url: str) -> str:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.pdf")
        _download_file(pdf_url, path)
        return extract_markdown_from_pdf(path)


def _extract_text_from_html_worker(html_url: str) -> str | None:
    import trafilatura

    downloaded = trafilatura.fetch_url(html_url)
    if downloaded is None:
        raise ValueError(f"Failed to download HTML from {html_url}")
    text = trafilatura.extract(downloaded, include_comments=False, include_tables=False)
    if not text:
        raise ValueError(f"No text extracted from {html_url}")
    return text


def _extract_text_from_tar_worker(source_url: str, paper_id: str, paper_title: str | None = None) -> str | None:
    with TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "paper.tar.gz")
        _download_file(source_url, path)
        file_contents = extract_tex_code_from_tar(path, paper_id, paper_title=paper_title)
        if not file_contents or "all" not in file_contents:
            raise ValueError("Main tex file not found.")
        return file_contents["all"]


# ---------------------------------------------------------------------------
# RSS fallback: build arxiv.Result objects directly from the RSS feed entries,
# used when export.arxiv.org/api/query rejects requests (e.g. HTTP 406 on
# GitHub Actions runners).
# ---------------------------------------------------------------------------

def _clean_rss_abstract(summary: str) -> str:
    # RSS summary looks like: "arXiv:2609.25031v1 Announce Type: new \nAbstract: ..."
    if "Abstract:" in summary:
        summary = summary.split("Abstract:", 1)[1]
    return re.sub(r"\s+", " ", summary).strip()


def _parse_rss_authors(entry: Any) -> list[ArxivResult.Author]:
    raw = entry.get("author", "") or ""
    if not raw and entry.get("authors"):
        raw = ", ".join(a.get("name", "") for a in entry.authors)
    names = [n.strip() for n in re.split(r",|\band\b", raw) if n.strip()]
    return [ArxivResult.Author(name) for name in names]


def _parse_rss_time(struct_time: Any) -> datetime:
    if struct_time:
        try:
            return datetime.fromtimestamp(calendar.timegm(struct_time), tz=timezone.utc)
        except Exception:
            pass
    return datetime.now(timezone.utc)


def _rss_entry_to_result(paper_id: str, entry: Any) -> ArxivResult:
    categories = [t.get("term") for t in entry.get("tags", []) if t.get("term")]
    published = _parse_rss_time(entry.get("published_parsed"))
    updated = _parse_rss_time(entry.get("updated_parsed") or entry.get("published_parsed"))
    title = re.sub(r"\s+", " ", entry.get("title", "")).strip()
    pdf_url = f"https://arxiv.org/pdf/{paper_id}"
    return ArxivResult(
        entry_id=f"http://arxiv.org/abs/{paper_id}",
        updated=updated,
        published=published,
        title=title,
        authors=_parse_rss_authors(entry),
        summary=_clean_rss_abstract(entry.get("summary", "")),
        primary_category=categories[0] if categories else "",
        categories=categories,
        links=[
            ArxivResult.Link(f"https://arxiv.org/abs/{paper_id}", title="abs", rel="alternate"),
            ArxivResult.Link(pdf_url, title="pdf", rel="related"),
        ],
    )


@register_retriever("arxiv")
class ArxivRetriever(BaseRetriever):
    def __init__(self, config):
        super().__init__(config)
        if self.config.source.arxiv.category is None:
            raise ValueError("category must be specified for arxiv.")

    def _retrieve_raw_papers(self) -> list[ArxivResult]:
        # Fewer internal retries so we fall back to RSS quickly when blocked.
        client = arxiv.Client(num_retries=2, delay_seconds=5)
        query = '+'.join(self.config.source.arxiv.category)
        include_cross_list = self.config.source.arxiv.get("include_cross_list", False)
        # Get the latest paper from arxiv rss feed
        feed = feedparser.parse(f"https://rss.arxiv.org/atom/{query}")
        if 'Feed error for query' in feed.feed.get("title", ""):
            raise Exception(f"Invalid ARXIV_QUERY: {query}.")
        raw_papers = []
        allowed_announce_types = {"new", "cross"} if include_cross_list else {"new"}
        id_to_entry = {}
        for entry in feed.entries:
            if entry.get("arxiv_announce_type", "new") not in allowed_announce_types:
                continue
            pid = entry.id.removeprefix("oai:arXiv.org:")
            id_to_entry.setdefault(pid, entry)
        all_paper_ids = list(id_to_entry.keys())
        if self.config.executor.debug:
            all_paper_ids = all_paper_ids[:10]

        # Get full information of each paper from arxiv api,
        # falling back to RSS metadata if the API refuses us.
        bar = tqdm(total=len(all_paper_ids))
        max_batch_retries = 3
        batch_retry_delay = 30
        use_rss = False
        for i in range(0, len(all_paper_ids), 20):
            batch_ids = all_paper_ids[i:i + 20]
            if not use_rss:
                search = arxiv.Search(id_list=batch_ids)
                for attempt in range(max_batch_retries):
                    try:
                        batch = list(client.results(search))
                        bar.update(len(batch))
                        raw_papers.extend(batch)
                        break
                    except arxiv.HTTPError as exc:
                        if exc.status == 429 and attempt < max_batch_retries - 1:
                            wait = batch_retry_delay * (attempt + 1)
                            logger.warning(f"arXiv API 429 on batch {i // 20}, retry {attempt + 1}/{max_batch_retries} in {wait}s")
                            sleep(wait)
                        else:
                            logger.warning(
                                f"arXiv API failed with HTTP {exc.status}; "
                                "falling back to RSS metadata for remaining papers."
                            )
                            use_rss = True
                            break
                    except Exception as exc:
                        logger.warning(f"arXiv API error ({exc}); falling back to RSS metadata.")
                        use_rss = True
                        break
            if use_rss:
                for pid in batch_ids:
                    try:
                        raw_papers.append(_rss_entry_to_result(pid, id_to_entry[pid]))
                    except Exception as exc:
                        logger.warning(f"Failed to build paper {pid} from RSS: {exc}")
                bar.update(len(batch_ids))
            elif i + 20 < len(all_paper_ids):
                sleep(3)
        bar.close()

        return raw_papers

    def convert_to_paper(self, raw_paper: ArxivResult) -> Paper:
        title = raw_paper.title
        authors = [a.name for a in raw_paper.authors]
        abstract = raw_paper.summary
        pdf_url = raw_paper.pdf_url
        full_text = extract_text_from_tar(raw_paper)
        if full_text is None:
            full_text = extract_text_from_html(raw_paper)
        if full_text is None:
            full_text = extract_text_from_pdf(raw_paper)
        return Paper(
            source=self.name,
            title=title,
            authors=authors,
            abstract=abstract,
            url=raw_paper.entry_id,
            pdf_url=pdf_url,
            full_text=full_text,
        )


def extract_text_from_html(paper: ArxivResult) -> str | None:
    html_url = paper.entry_id.replace("/abs/", "/html/")
    try:
        return _extract_text_from_html_worker(html_url)
    except Exception as exc:
        logger.warning(f"HTML extraction failed for {paper.title}: {exc}")
        return None


def extract_text_from_pdf(paper: ArxivResult) -> str | None:
    if paper.pdf_url is None:
        logger.warning(f"No PDF URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_pdf_worker,
        (paper.pdf_url,),
        timeout=PDF_EXTRACT_TIMEOUT,
        operation="PDF extraction",
        paper_title=paper.title,
    )


def extract_text_from_tar(paper: ArxivResult) -> str | None:
    source_url = paper.source_url()
    if source_url is None:
        logger.warning(f"No source URL available for {paper.title}")
        return None
    return _run_with_hard_timeout(
        _extract_text_from_tar_worker,
        (source_url, paper.entry_id, paper.title),
        timeout=TAR_EXTRACT_TIMEOUT,
        operation="Tar extraction",
        paper_title=paper.title,
    )
