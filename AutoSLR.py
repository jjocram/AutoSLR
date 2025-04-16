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

    query = mo.ui.text(placeholder="Search...", label="Filter: ", value=q1, full_width=True)

    mo.vstack([query])
    return q1, q2, query


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
    page_limit = 500
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
        print(el)
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
            pl.from_dicts(map(create_table_row, map(extract_DOI, details_request)))
        ])

        citation_df = pl.concat([
            citation_df,
            pl.from_dicts(map(create_table_row, 
                              map(extract_DOI, 
                                  itertools.chain.from_iterable(map(lambda el: el["citations"], details_request)))))
        ])
    
        reference_df = pl.concat([
            citation_df,
            pl.from_dicts(map(create_table_row, 
                              map(extract_DOI, 
                                  itertools.chain.from_iterable(map(lambda el: el["references"], details_request)))))
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
def _(citation_df, mo, pl):
    mo.vstack([mo.md("### Filtered dataset"), citation_df.filter((pl.col("title").str.contains_any(["bpmn", "simulation", "manufacturing"])) | (pl.col("abstract").str.contains_any(["bpmn", "simulation", "manufacturing"])))])
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
def _(mo, pl, reference_df):
    mo.vstack([mo.md("### Filtered dataset"), reference_df.filter((pl.col("title").str.contains_any(["bpmn", "simulation", "manufacturing"])) | (pl.col("abstract").str.contains_any(["bpmn", "simulation", "manufacturing"])))]) 
    return


if __name__ == "__main__":
    app.run()
