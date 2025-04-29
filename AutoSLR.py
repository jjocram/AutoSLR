import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import polars as pl
    import itertools
    import pyalex as alex
    from datetime import datetime
    import hashlib
    return alex, datetime, hashlib, itertools, mo, pl, requests


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Parameters

        ## Query: 
        The match query supports the following syntax:

        - `+` for AND operation
        - `|` for OR operation
        - `-` negates a term
        - `"` collects terms into a phrase
        - `*` can be used to match a prefix
        - `(` and `)` for precedence
        - `~N` after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
        - `~N` after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)
        """
    )
    return


@app.cell
def _(mo):
    q1 = "(BPMN + manufacturing) | (manufacturing + simulation) | (BPMN + simulation)"
    q2 = "BPMN + manufacturing + simulation"

    query = mo.ui.text(placeholder="Search...", label="Search words: ", value=q1, full_width=True)

    mo.vstack([query])
    return q1, q2, query


@app.cell(hide_code=True)
def _(mo, query):
    import re
    words = set(word.lower() for word in re.findall(r'\b\w+\b', query.value))

    base_cit = mo.ui.number(start=0, label="Citation base requirements", value=2, full_width=True)
    cit_per_year = mo.ui.number(start=0, label="Citations required per year of age", full_width=True)

    remove_no_doi = mo.ui.switch(label="Remove results with no DOI", value=True)

    terms_to_filter = mo.ui.array(elements= [mo.ui.text(placeholder="Term", value=term) for term in words], 
                                  label="Terms to filter")

    mo.vstack([base_cit, cit_per_year, remove_no_doi, terms_to_filter], align="start")
    return base_cit, cit_per_year, re, remove_no_doi, terms_to_filter, words


@app.cell
def _(
    alex,
    base_cit,
    cit_per_year,
    datetime,
    hashlib,
    itertools,
    mo,
    pl,
    remove_no_doi,
    requests,
    terms_to_filter,
):
    from typing import List, Tuple, Dict
    from pathlib import Path 
    import json

    class PaperData:
        @property
        def columns(self) -> Dict[str, str]:
            raise NotImplementedError("""
            You need to declare a dictionary which maps specific names to yours column names: 
                - title
                - DOI
                - abstract
                - year
                - citationCount""")

        @property
        def use_abstract_inverted_index(self) -> bool:
            raise NotImplementedError

        def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
            CURRENT_YEAR = datetime.now().year
            ALPHA = cit_per_year.value  # Citations required per year of age
            BETA = base_cit.value   # Base requirement
            filtered_df = df

            if remove_no_doi.value:
                filtered_df = filtered_df.filter(pl.col(self.columns["DOI"]).is_not_null())

            terms = terms_to_filter.value

            if self.use_abstract_inverted_index:
                condition_contains_terms = (
                    (pl.col(self.columns["title"]).str.contains_any(terms_to_filter.value)) |
                    (pl.col(self.columns["abstract"])
                         .map_elements(lambda x: any(term in x for term in terms_to_filter.value), 
                                       return_dtype=pl.Boolean))
                )
            else:
                condition_contains_terms = (
                    pl.col(self.columns["title"]).str.contains_any(terms) |
                    pl.col(self.columns["abstract"]).str.contains_any(terms)
                )

            condition = pl.col(self.columns["abstract"]).is_null() | condition_contains_terms
            filtered_df = filtered_df.filter(condition)

            # Step 1: Add age and min required citations
            filtered_df = filtered_df.with_columns(
                age=(CURRENT_YEAR - pl.col(self.columns["year"])),
                required_citations=(ALPHA * (CURRENT_YEAR - pl.col(self.columns["year"])) + BETA)
            )

            # Step 2: Filter by actual citation count
            filtered_df = filtered_df.filter(
                pl.col(self.columns["citationCount"]) >= pl.col("required_citations")
            )

            # Optional: sort by citationCount or age-adjusted score
            filtered_df = filtered_df.sort(self.columns["citationCount"], descending=True)

            return filtered_df

        @property
        def filtered_references_df(self) -> pl.DataFrame:
            assert self.reference_df is not None, "References dataframe is not available"
            return self._apply_filters(self.reference_df.unique(self.columns["DOI"]))

        @property
        def filtered_citations_df(self)-> pl.DataFrame:
            assert self.citation_df is not None, "Citations dataframe is not available"
            return self._apply_filters(self.citation_df.unique(self.columns["DOI"]))

        @property
        def filtered_search_df(self) -> pl.DataFrame:
            assert self.citation_df is not None, "Searched dataframe is not available"
            return self._apply_filters(self.search_df.unique(self.columns["DOI"]))


    class SemanticScholarData(PaperData):
        def _fetch_ids(self, search_query: str) -> List[str]:
            ids =[]

            url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={search_query}"
            id_request = requests.get(url).json()

            with mo.status.spinner(subtitle="Searching papers..."):
                while True:
                    if "data" in id_request:
                        for paper in id_request["data"]:
                            ids.append(paper["paperId"])

                    if "token" not in id_request:
                        break

                    id_request = requests.get(f"{url}&token={id_request['token']}").json()

            with open(self.cache_path.joinpath("ids.txt"), "w+") as ids_file:
                ids_file.writelines([f"{id}\n" for id in ids])

            return ids

        def _fetch_dataframes(self, ids) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            page_limit = 100

            base_fields = ["abstract", 
                           "venue", 
                           "year", 
                           "fieldsOfStudy", 
                           "citationCount", 
                           "externalIds", 
                           "publicationTypes"
                          ]
            references_fields = [f"references.{field}" for field in base_fields]
            citation_fields = [f"citations.{field}" for field in base_fields]
            all_fields = base_fields + references_fields + citation_fields

            table_schema = {
                "paperId": pl.String, 
                "DOI": pl.String,
                "title": pl.String, 
                "abstract": pl.String, 
                "venue": pl.String,
                "year": pl.Int64,
                "fieldsOfStudy": pl.List(pl.String),
                "citationCount": pl.Int64, 
                "pubblicationTypes": pl.List(pl.String),
            }
            search_df = pl.DataFrame(schema=table_schema)
            citation_df = pl.DataFrame(schema=table_schema)
            reference_df = pl.DataFrame(schema=table_schema)

            def extract_DOI(el: dict) -> dict:
                try:
                    return el | {"DOI": el.get("externalIds", {}).get("DOI") if isinstance(el.get("externalIds"), dict) else None}
                except:
                    print("Exception raised while looking for DOI in", el)
                    return {"paperId": None, 
                            "DOI": None,
                            "title": None, 
                            "abstract": None, 
                            "venue": None,
                            "year": None,
                            "fieldsOfStudy": None,
                            "citationCount": None, 
                            "pubblicationTypes": None
                           }


            def create_table_row(el: dict) -> dict:
                return {k: el.get(k, None) for k in table_schema.keys()}

            for page_ids in mo.status.progress_bar([ids[i:i+page_limit] for i in range(0, len(ids), page_limit)], 
                                                   title="Fetching papers...", 
                                                   completion_title=f"Fetch complete"):
                details_request = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": ",".join(all_fields)},
                    json={"ids": page_ids}
                ).json()
                mo.stop("error" in details_request, output=details_request)

                search_df = pl.concat([
                    search_df,
                    pl.from_dicts(map(create_table_row, map(extract_DOI, details_request)), 
                                  infer_schema_length=None)])

                citations = [el["citations"] for el in details_request if isinstance(el, dict)]
                flat_citations = itertools.chain.from_iterable(citations)
                flat_citation_with_doi = [extract_DOI(el) for el in flat_citations]
                flat_citation_with_doi_as_dict = [create_table_row(el) for el in flat_citation_with_doi]
                citation_df = pl.concat([citation_df,
                                         pl.from_dicts(flat_citation_with_doi_as_dict, infer_schema_length=None)])

                references = [el["references"] for el in details_request if isinstance(el, dict)]
                flat_references = itertools.chain.from_iterable(references)
                flat_references_with_doi = [extract_DOI(el) for el in flat_references]
                flat_references_with_doi_as_dict = [create_table_row(el) for el in flat_citation_with_doi]
                if len(flat_references_with_doi_as_dict) > 0:
                    reference_df = pl.concat([citation_df, 
                                              pl.from_dicts(flat_citation_with_doi_as_dict, infer_schema_length=None)])

            return search_df, citation_df, reference_df

        def __init__(self, search_query: str):
            self.cache_path = Path(f"cache_datasets/semantic_scholar/d{hashlib.md5(search_query.encode("utf-8")).hexdigest()}")
            self.cache_path.mkdir(parents=True, exist_ok=True)

            is_cache_available = {"search.json", "citation.json", "reference.json"}.issubset({f.parts[-1] for f in self.cache_path.iterdir()})

            if is_cache_available:
                self.search_df = pl.read_json(self.cache_path.joinpath("search.json"))
                self.citation_df = pl.read_json(self.cache_path.joinpath("citation.json"))
                self.reference_df = pl.read_json(self.cache_path.joinpath("reference.json"))
            else:
                if "ids.txt" in [f.parts[-1] for f in self.cache_path.iterdir()]:
                    with open(self.cache_path.joinpath("ids.txt"), "r") as ids_file:
                        ids = ids_file.read().splitlines()
                else:
                    ids = self._fetch_ids(search_query)
                self.search_df, self.citation_df, self.reference_df = self._fetch_dataframes(ids)

                with open(self.cache_path.joinpath("search.json"), "w+") as f:
                    self.search_df.write_json(f)

                with open(self.cache_path.joinpath("citation.json"), "w+") as f:
                    self.citation_df.write_json(f)

                with open(self.cache_path.joinpath("reference.json"), "w+") as f:
                    self.reference_df.write_json(f)

        @property
        def columns(self) -> Dict[str, str]:
            return {
                "DOI": "DOI",
                "citationCount": "citationCount",
                "title": "title",
                "abstract": "abstract",
                "year": "year"
            }

        @property
        def use_abstract_inverted_index(self) -> bool:
            return False


    class OpenAlexData(PaperData):
        def _fetch_dataframes(self, search_query: str) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
            page_limit = 50
            fields = ["id", 
                      "doi", 
                      "title", 
                      "publication_year", 
                      "type", 
                      "primary_location", 
                      "cited_by_count", 
                      "abstract_inverted_index"]

            search = alex.Works().search(search_query).select(fields + ["referenced_works"]).get()
            paper_ids = {p["id"].split("/")[-1] for p in search}

            citations_search = alex.Works().filter(cites="|".join(paper_ids)).select(fields).get()

            ref_ids = list(set(itertools.chain.from_iterable([p["referenced_works"] for p in search])))
            references_search = itertools.chain.from_iterable(
                [alex.Works().filter(openalex="|".join(ids)).select(fields).get() 
                 for ids in mo.status.progress_bar([ref_ids[i:i+page_limit] for i in range(0, len(ref_ids), page_limit)], 
                                                   title="Fetching references...", 
                                                   completion_title=f"Fetch complete")]
            )

            def extract_venue(p: dict) -> dict:
                return {"venue": p.get("primary_location").get("source").get("display_name", None) if isinstance(p.get("primary_location"), dict) and isinstance(p["primary_location"].get("source"), dict) else None}

            alex_search_df = pl.from_dicts([ p | extract_venue(p) for p in search]).drop("referenced_works", "primary_location")
            alex_citations_df = pl.DataFrame([ p | extract_venue(p) for p in citations_search]).drop("primary_location")
            alex_references_df = pl.from_dicts([p | extract_venue(p) for p in references_search]).drop("primary_location")

            return alex_search_df, alex_citations_df, alex_references_df

        def __init__(self, search_query: str):
            self.cache_path = Path(f"cache_datasets/open_alex/d{hashlib.md5(search_query.encode("utf-8")).hexdigest()}")

            if self.cache_path.exists():
                self.search_df = pl.read_json(self.cache_path.joinpath("search.json"))
                self.citation_df = pl.read_json(self.cache_path.joinpath("citation.json"))
                self.reference_df = pl.read_json(self.cache_path.joinpath("reference.json"))
            else:
                self.search_df, self.citation_df, self.reference_df = self._fetch_dataframes(search_query)

                self.cache_path.mkdir(parents=True, exist_ok=True)

                with open(self.cache_path.joinpath("search.json"), "w+") as f:
                    self.search_df.write_json(f)

                with open(self.cache_path.joinpath("citation.json"), "w+") as f:
                    self.citation_df.write_json(f)

                with open(self.cache_path.joinpath("reference.json"), "w+") as f:
                    self.reference_df.write_json(f)

        @property
        def columns(self) -> Dict[str, str]:
            return {
                "DOI": "doi",
                "citationCount": "cited_by_count",
                "title": "title",
                "abstract": "abstract_inverted_index",
                "year": "publication_year"
            }

        @property
        def use_abstract_inverted_index(self) -> bool:
            return True

        @property
        def filtered_references_df(self) -> pl.DataFrame:
            return self._apply_filters(self.reference_df)

        @property
        def filtered_citations_df(self) -> pl.DataFrame:
            return self._apply_filters(self.citation_df)

        @property
        def filtered_search_df(self) -> pl.DataFrame:
            self._apply_filters(self.search_df)
    return (
        Dict,
        List,
        OpenAlexData,
        PaperData,
        Path,
        SemanticScholarData,
        Tuple,
        json,
    )


@app.cell
def _(SemanticScholarData, query):
    data = SemanticScholarData(query.value)
    # data = OpenAlexData(" AND ".join(words))
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Results""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from search""")
    return


@app.cell(hide_code=True)
def _(data, mo, pl):
    mo.vstack([mo.md("### Full dataset"), data.search_df.filter(pl.col("DOI").is_not_null()).select("title", "DOI")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from all citations""")
    return


@app.cell(hide_code=True)
def _(data, mo):
    mo.vstack([mo.md("### Full dataset"), data.citation_df])
    return


@app.cell(hide_code=True)
def _(data, mo):
    mo.vstack([mo.md("### Filtered dataset"), data.filtered_citations_df.select("title", "DOI")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from all references""")
    return


@app.cell(hide_code=True)
def _(data, mo):
    mo.vstack([mo.md("### Full dataset"), data.reference_df])
    return


@app.cell(hide_code=True)
def _(data, mo):
    mo.vstack([mo.md("### Filtered dataset"), data.filtered_references_df.select("title", "DOI")]) 
    return


if __name__ == "__main__":
    app.run()
