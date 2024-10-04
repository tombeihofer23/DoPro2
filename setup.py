"""Setup project."""

import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "DoPro2"
AUTHOR_USER_NAME = ""
SCR_REPO = "dopro2_HEFTcom_challenge"
AUTHOR_EMAIL = ""


setuptools.setup(
    name=SCR_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="DoPro2 Hybrid Energy Forecasting and"
    "Trading Competition, Team Tomami",
    long_description=long_description,
    url=f"https://github.com/tombeihofer23/{REPO_NAME}"
)
