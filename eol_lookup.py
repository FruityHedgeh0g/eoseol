#!/usr/bin/env python3
"""Recherche le statut de fin de vie (End of Life) d'une technologie.

Stratégie :
1. Interrogation de l'API officielle https://endoflife.date/docs/api/v1/
   (gratuite, sans clé). Cette source fait foi lorsqu'elle est concluante.
2. À défaut de réponse concluante (produit ou cycle absent du référentiel),
   bascule sur un agent LangChain outillé d'une recherche DuckDuckGo (gratuite)
   et d'une récupération de contenu via curl, chargé de trouver l'information
   sur le Web.

Usage :
    python eol_lookup.py java --cycle 21
    python eol_lookup.py postgresql --cycle 16
    python eol_lookup.py angular
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from typing import Any

import requests
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

ENDOFLIFE_API_BASE = "https://endoflife.date/api/v1"
ANTHROPIC_MODEL = "claude-opus-5"


def query_endoflife_api(product: str, cycle: str | None) -> dict[str, Any] | None:
    """Interroge endoflife.date et renvoie le statut EOL, ou None si non concluant."""
    try:
        response = requests.get(f"{ENDOFLIFE_API_BASE}/products/{product}", timeout=10)
    except requests.RequestException:
        return None

    if response.status_code != 200:
        return None

    result = response.json().get("result")
    if not result:
        return None

    releases = result.get("releases", [])
    if cycle is None:
        release = releases[0] if releases else None
    else:
        release = next(
            (r for r in releases if cycle in (r.get("name"), r.get("label"))),
            None,
        )

    # isEol absent => le référentiel ne sait pas trancher pour ce cycle.
    if release is None or release.get("isEol") is None:
        return None

    return {
        "source": "endoflife.date",
        "product": product,
        "cycle": release.get("name"),
        "isEol": release.get("isEol"),
        "eolFrom": release.get("eolFrom"),
        "isMaintained": release.get("isMaintained"),
        "url": f"https://endoflife.date/{product}",
    }


@tool
def duckduckgo_search(query: str) -> str:
    """Recherche `query` sur DuckDuckGo et renvoie les résultats (titre, lien, extrait) en JSON."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=5))
    return json.dumps(results, ensure_ascii=False)


@tool
def fetch_url_content(url: str) -> str:
    """Récupère le contenu brut d'une URL via curl et le renvoie (tronqué à 5000 caractères)."""
    try:
        completed = subprocess.run(
            ["curl", "-sL", "--max-time", "10", url],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return f"Erreur : délai dépassé lors de la récupération de {url}"

    if completed.returncode != 0:
        return f"Erreur curl (code {completed.returncode}) pour {url}"

    return completed.stdout[:5000]


def run_web_fallback_agent(product: str, cycle: str | None) -> str:
    """Fait chercher la date de fin de vie sur le Web par un agent LangChain/Claude."""
    llm = ChatAnthropic(model=ANTHROPIC_MODEL)
    tools = [duckduckgo_search, fetch_url_content]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Tu es un assistant chargé de déterminer le statut de fin de vie "
                "(End of Life) d'une technologie logicielle. Utilise l'outil "
                "duckduckgo_search pour repérer des sources pertinentes (documentation "
                "officielle en priorité), puis fetch_url_content pour lire le contenu "
                "des pages les plus prometteuses avant de conclure. Indique toujours "
                "la source retenue et, si l'information reste introuvable, dis-le "
                "explicitement plutôt que de deviner.",
            ),
            ("human", "{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    query = f"Date de fin de vie (End of Life) de {product}"
    if cycle:
        query += f" version {cycle}"

    result = executor.invoke({"input": query})
    return result["output"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("product", help="Identifiant du produit (ex. 'java', 'postgresql', 'angular')")
    parser.add_argument("--cycle", default=None, help="Cycle/version précis (ex. '21', '16')")
    args = parser.parse_args()

    api_result = query_endoflife_api(args.product, args.cycle)
    if api_result is not None:
        print(json.dumps(api_result, ensure_ascii=False, indent=2))
        return

    print(
        "Aucune réponse concluante sur endoflife.date, bascule vers la recherche web...",
        file=sys.stderr,
    )
    print(run_web_fallback_agent(args.product, args.cycle))


if __name__ == "__main__":
    main()
