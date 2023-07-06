import os
import aiohttp
import asyncio
import threading
import streamlit as st
from pathlib import Path
from typing import List
from tqdm.auto import tqdm


def make_clickable(link):
    return f'<a target="_blank" href="{link}">link</a>'


def hide_header_and_footer():
    hide_streamlit_style = """
        <style>
            #MainMenu {visibility: hidden;}
            header {visibility: hidden;}
            footer {visibility: hidden;}
        </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)


def read_markdown_file(markdown_file):
    """
    Examples
    --------
    >>> markdown = read_markdown_file("introduction.md")
    >>> st.markdown(markdown, unsafe_allow_html=True)
    """
    return Path(markdown_file).read_text(encoding='utf-8')


def read_css_file(css_file):
    """
    Examples
    --------
    >>> css_file = read_markdown_file("main.css")
    >>> st.markdown(css_file, unsafe_allow_html=True)
    """
    return f"<style>{Path(css_file).read_text(encoding='utf-8')}</style>"


def fast_scandir(path: str, exts: List[str], recursive: bool = False):
    """
    Scan files recursively faster than glob
    Credit from github.com/drscotthawley/aeiou/blob/main/aeiou/core.py
    """
    subfolders, files = [], []

    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(path):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    if os.path.splitext(f.name)[1].lower() in exts:
                        files.append(f.path)
            except Exception:
                pass
    except Exception:
        pass

    if recursive:
        for path in list(subfolders):
            sf, f = fast_scandir(path, exts, recursive=recursive)
            subfolders.extend(sf)
            files.extend(f)  # type: ignore

    return subfolders, files


class RunThread(threading.Thread):

    def __init__(self, func):
        self.func = func
        self.result = None
        super().__init__()

    def run(self):
        self.result = asyncio.run(self.func)


def run_async(func):
    """Allows to run asyncio in an already running loop, e.g. Jupyter notebooks"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        thread = RunThread(func)
        thread.start()
        thread.join()
        return thread.result
    else:
        return asyncio.run(func)


class Downloader:
    """
    Credit from https://github.com/archinetai/audio-data-pytorch
    """
    def __init__(
        self,
        urls: List[str],
        path: str = ".",
        remove_on_exit: bool = False,
        check_exists: bool = True,
        description: str = "Downloading",
    ):
        self.urls = urls
        self.path = path
        self.files: List[str] = []
        self.remove_on_exit = remove_on_exit
        self.check_exists = check_exists
        self.description = description

    def get_file_path(self, url: str) -> str:
        os.makedirs(self.path, exist_ok=True)
        filename = url.split("/")[-1]
        return os.path.join(self.path, filename)

    async def download(self, url: str, session):

        async with session.get(url) as response:
            file_path = self.get_file_path(url)
            total_size_in_bytes = int(response.headers.get("content-length", 0))
            chunk_size = 1024

            progress_bar = tqdm(
                desc=f"{self.description}: {file_path}",
                total=total_size_in_bytes,
                unit="iB",
                unit_scale=True,
                unit_divisor=1024,
            )

            with open(file_path, "wb") as file:
                async for chunk in response.content.iter_chunked(chunk_size):
                    size = file.write(chunk)
                    progress_bar.update(size)

            return file_path

    async def download_all_async(self) -> List[str]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None)  # Unlimited timeout time
        ) as session:
            tasks = []
            for url in self.urls:
                file_path = self.get_file_path(url)
                # Check if file already exists
                if self.check_exists and os.path.exists(file_path):
                    self.files += [file_path]
                else:
                    tasks += [self.download(url, session)]
            self.files += await asyncio.gather(*tasks, return_exceptions=True)
        return self.files

    def download_all(self) -> List[str]:
        return run_async(self.download_all_async())

    def remove_files(self):
        for file in self.files:
            os.remove(file)

    def __enter__(self):
        return self.download_all()

    def __exit__(self, *args):
        if self.remove_on_exit:
            self.remove_files()

    async def __aenter__(self):
        return await self.download_all_async()

    async def __aexit__(self, *args):
        if self.remove_on_exit:
            self.remove_files()