import marimo

__generated_with = "0.12.9"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import requests
    import polars as pl
    return mo, pl, requests


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
    q2 = "BPMn + manufacturing + simulation"

    query = mo.ui.text(placeholder="Search...", label="Search words: ", value=q2, full_width=True)

    mo.vstack([query])
    return q1, q2, query


@app.cell
def _(mo, query):
    import re
    words = set(word.lower() for word in re.findall(r'\b\w+\b', query.value))

    min_citations = mo.ui.number(start=0, label="Minimum citations", value=10, full_width=True)

    remove_no_doi = mo.ui.switch(label="Remove results with no DOI", value=True)

    terms_to_filter = mo.ui.array(elements= [mo.ui.text(placeholder="Term", value=term) for term in words], 
                                  label="Terms to filter")

    mo.vstack([min_citations, remove_no_doi, terms_to_filter], align="start")
    return min_citations, re, remove_no_doi, terms_to_filter, words


@app.cell
def _(query, requests):
    url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={query.value}"
    id_request = requests.get(url).json()
    print(f"Will retrieve an estimated {id_request['total']} documents")

    ids = []

    while True:
        if "data" in id_request:
            for paper in id_request["data"]:
                ids.append(paper["paperId"])

        if "token" not in id_request:
            break

        id_request = requests.get(f"{url}&token={id_request['token']}").json()
    return id_request, ids, paper, url


@app.cell
def _(ids, mo, pl, requests):
    import itertools
    import json
    page_limit = 250
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

    search_df = pl.DataFrame(schema=table_schema)
    citation_df = pl.DataFrame(schema=table_schema)
    reference_df = pl.DataFrame(schema=table_schema)

    def extract_DOI(el: dict) -> dict:
        return el | {"DOI": el.get("externalIds", {}).get("DOI") if isinstance(el.get("externalIds"), dict) else None}

    def create_table_row(el: dict) -> dict:
        return {k: el.get(k, None) for k in table_schema.keys()}

    for page_ids in mo.status.progress_bar([ids[i:i+page_limit] for i in range(0, len(ids), page_limit)]):
        details_request = requests.post(
            "https://api.semanticscholar.org/graph/v1/paper/batch",
            params={"fields": ",".join(all_fields)},
            json={"ids": page_ids}
        ).json()
        mo.stop("error" in details_request, output=details_request)

        search_df = pl.concat([
            search_df,
            pl.from_dicts(map(create_table_row, map(extract_DOI, details_request)),
                          infer_schema_length=None)
        ])

        citation_df = pl.concat([
            citation_df,
            pl.from_dicts(map(create_table_row, 
                              map(extract_DOI, 
                                  itertools.chain.from_iterable(map(lambda el: el["citations"], details_request)))),
                          infer_schema_length=None)
        ])

        reference_df = pl.concat([
            citation_df,
            pl.from_dicts(map(create_table_row, 
                              map(extract_DOI, 
                                  itertools.chain.from_iterable(map(lambda el: el["references"], details_request)))),
                          infer_schema_length=None)
        ])

    return (
        all_fields,
        base_fields,
        citation_df,
        citation_fields,
        create_table_row,
        details_request,
        extract_DOI,
        itertools,
        json,
        page_ids,
        page_limit,
        reference_df,
        references_fields,
        search_df,
        table_schema,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Results""")
    return


@app.cell
def _(min_citations, pl, remove_no_doi, terms_to_filter):
    def apply_filters(df: pl.DataFrame) -> pl.DataFrame:
        filterd_df = df.filter((pl.col("title").str.contains_any(terms_to_filter.value)) |
                                              (pl.col("abstract").str.contains_any(terms_to_filter.value))
                                             )

        filterd_df = filterd_df.filter(pl.col("citationCount") >= min_citations.value)

        if remove_no_doi.value:
            filterd_df = filterd_df.filter(pl.col("DOI").is_not_null())

        return filterd_df
    return (apply_filters,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from search""")
    return


@app.cell(hide_code=True)
def _(mo, search_df):
    mo.vstack([mo.md("### Full dataset"), search_df])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from all citations""")
    return


@app.cell(hide_code=True)
def _(citation_df, mo):
    mo.vstack([mo.md("### Full dataset"), citation_df])
    return


@app.cell(hide_code=True)
def _(apply_filters, citation_df, mo):
    mo.vstack([mo.md("### Filtered dataset"), apply_filters(citation_df)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Dataframe from all references""")
    return


@app.cell(hide_code=True)
def _(mo, reference_df):
    mo.vstack([mo.md("### Full dataset"), reference_df])
    return


@app.cell(hide_code=True)
def _(apply_filters, mo, reference_df):
    mo.vstack([mo.md("### Filtered dataset"), apply_filters(reference_df)]) 
    return


if __name__ == "__main__":
    app.run()
