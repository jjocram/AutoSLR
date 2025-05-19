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
    from sentence_transformers import SentenceTransformer
    return (
        SentenceTransformer,
        alex,
        datetime,
        hashlib,
        itertools,
        mo,
        pl,
        requests,
    )


@app.cell
def _(SentenceTransformer):
    import spacy
    from sentence_transformers.cross_encoder import CrossEncoder


    nlp = spacy.load("en_core_web_lg")
    sbert_model = SentenceTransformer('sentence-transformers/allenai-specter')

    # Preprocessing (lemmatization + punctuation removal)
    def preprocess(sentence):
        doc = nlp(sentence.lower())
        return ' '.join([token.lemma_ for token in doc if token.is_alpha])
    return CrossEncoder, nlp, preprocess, sbert_model, spacy


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Parameters

        ## Query: 
        The match query supports the following syntax:

        - ++ for AND operation
        - ∣| for OR operation
        - -- negates a term
        - "" collects terms into a phrase
        - ⋅* can be used to match a prefix
        - (( and )) for precedence
        - ~N~N after a word matches within the edit distance of N (Defaults to 2 if N is omitted)
        - ~N~N after a phrase matches with the phrase terms separated up to N terms apart (Defaults to 2 if N is omitted)
        """
    )
    return


@app.cell
def _(mo):
    bpmn = "BPMN"
    manufacturing = "manufacturing"
    simulation = "simulation"

    q1 = f"({bpmn} + {manufacturing}) | ({manufacturing} + {simulation}) | ({bpmn} + {simulation})"
    q2 = f"{bpmn} + {manufacturing} + {simulation}"
    q3 = f"({bpmn} + {manufacturing}) | ({simulation} + {manufacturing})"

    query = mo.ui.text(placeholder="Search...", label="Search words: ", value=q2, full_width=True)

    mo.vstack([query])
    return bpmn, manufacturing, q1, q2, q3, query, simulation


@app.cell(hide_code=True)
def _(mo):
    base_cit = mo.ui.number(start=0, label="Citation base requirements", value=2, full_width=True)
    cit_per_year = mo.ui.number(start=0, label="Citations required per year of age", value=3, full_width=True)

    remove_no_doi = mo.ui.switch(label="Remove results with no DOI", value=True)

    mo.vstack([base_cit, cit_per_year, remove_no_doi], align="start")
    return base_cit, cit_per_year, remove_no_doi


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
):
    from typing import List, Tuple, Dict
    from pathlib import Path 
    import json
    import re

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

        def _apply_filters(self, df: pl.DataFrame) -> pl.DataFrame:
            CURRENT_YEAR = datetime.now().year
            ALPHA = cit_per_year.value  # Citations required per year of age
            BETA = base_cit.value   # Base requirement
            filtered_df = df

            if remove_no_doi.value:
                filtered_df = filtered_df.filter(pl.col(self.columns["DOI"]).is_not_null())

            filtered_df = filtered_df.filter(pl.col("fieldsOfStudy").list.contains("Computer Science") | pl.col("fieldsOfStudy").list.contains("Engineering")) 

            # Step 1: Add age and min required citations
            # filtered_df = filtered_df.with_columns(
            #     age=(CURRENT_YEAR - pl.col(self.columns["year"])),
            #     required_citations=(ALPHA * (CURRENT_YEAR - pl.col(self.columns["year"])) + BETA)
            # )

            # # Step 2: Filter by actual citation count
            # filtered_df = filtered_df.filter(
            #     pl.col(self.columns["citationCount"]) >= pl.col("required_citations")
            # )

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
            assert self.search_df is not None, "Searched dataframe is not available"
            assert not self.search_df.is_empty(), "Searched dataframe is empty"
            return self._apply_filters(self.search_df.unique(self.columns["DOI"]))

        @property
        def related_papers_df(self) -> pl.DataFrame:
            assert not self.citation_df.is_empty() and not self.reference_df.is_empty(), "Citation and reference are both empty"
            return pl.concat([self.citation_df, self.reference_df]).unique(self.columns["DOI"])

        @property
        def filtered_related_papers_df(self) -> pl.DataFrame:
            return self._apply_filters(self.related_papers_df)

        # def __init__(self):
        #     self.cache_path = Path(f"cache_datasets/{type(self).__name__}")
        #     self.cache_path.mkdir(parents=True, exist_ok=True)


    def is_a_paper_to_keep(paper: dict) -> bool:
        CURRENT_YEAR = datetime.now().year
        ALPHA = cit_per_year.value  # Citations required per year of age
        BETA = base_cit.value   # Base requirement

        if paper is None:
            return False

        if not isinstance(paper, dict):
            return False

        if "citationCount" not in paper or paper.get("citationCount", None) is None: 
            return False

        if "year" not in paper or paper.get("year", None) is None:
            return False

        if "externalIds" not in paper:
            return False

        if "DOI" not in paper["externalIds"]:
            return False

        # min_cit = ALPHA * (CURRENT_YEAR - paper["year"]) + BETA
        # if paper["citationCount"] < min_cit:
        #     return False

        return True

    class SemanticScholarData(PaperData):
        def _fetch_ids(self, search_query: str) -> List[str]:
            ids =[]

            url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={search_query}"
            id_request = requests.get(url).json()

            with mo.status.spinner(subtitle="Searching papers..."):
                while True:
                    mo.stop("error" in id_request, output=id_request)
                    if "data" in id_request:
                        for paper in id_request["data"]:
                            ids.append(paper["paperId"])

                    if "token" not in id_request or id_request["token"] is None:
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


            def extract_papers(papers: list[dict]) -> list[dict]:
                papers_with_doi = [extract_DOI(el) for el in papers]
                return [create_table_row(el) for el in papers_with_doi]

            def extract_subpapers(kind: str, search_result: list[dict]) -> list[dict]:
                papers = [el[kind] for el in search_result if isinstance(el, dict) and is_a_paper_to_keep(el)]
                flatten = itertools.chain.from_iterable(papers)
                return extract_papers(flatten)


            for page_ids in mo.status.progress_bar([ids[i:i+page_limit] for i in range(0, len(ids), page_limit)], 
                                                   title="Fetching papers...", 
                                                   completion_title=f"Fetch complete"):
                details_request = requests.post(
                    "https://api.semanticscholar.org/graph/v1/paper/batch",
                    params={"fields": ",".join(all_fields)},
                    json={"ids": page_ids}
                ).json()
                mo.stop("error" in details_request, output=details_request)

                search_papers = extract_papers([el for el in details_request if is_a_paper_to_keep(el)])
                if len(search_papers) > 0:
                    search_df = pl.concat([search_df, pl.from_dicts(search_papers, infer_schema_length=None)])

                citations = extract_subpapers("citations", details_request)
                if len(citations) > 0:
                    citation_df = pl.concat([citation_df, pl.from_dicts(citations, infer_schema_length=None)])

                references = extract_subpapers("references", details_request)
                if len(references) > 0:
                    reference_df = pl.concat([reference_df, pl.from_dicts(references, infer_schema_length=None)])

            return search_df, citation_df, reference_df

        def __init__(self, search_query: str):
            self.query_cache = hashlib.md5(search_query.encode("utf-8")).hexdigest()
            self.cache_path = Path(f"cache_datasets/semantic_scholar/d{self.query_cache}")
            self.cache_path.mkdir(parents=True, exist_ok=True)

            is_cache_available = {"search.json", "citation.json", "reference.json"}.issubset({f.parts[-1] for f in self.cache_path.iterdir()})

            if is_cache_available:
                print("Using cache:", self.query_cache)
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

                self.search_df.write_json(self.cache_path.joinpath("search.json"))
                self.citation_df.write_json(self.cache_path.joinpath("citation.json"))
                self.reference_df.write_json(self.cache_path.joinpath("reference.json"))

        @property
        def columns(self) -> Dict[str, str]:
            return {
                "DOI": "DOI",
                "citationCount": "citationCount",
                "title": "title",
                "abstract": "abstract",
                "year": "year"
            }



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
        is_a_paper_to_keep,
        json,
        re,
    )


@app.cell
def _(Dict, PaperData, pl):
    class JoinedPapers(PaperData):
        @property
        def columns(self) -> Dict[str, str]:
            return {
                "DOI": "DOI",
                "citationCount": "citationCount",
                "title": "title",
                "abstract": "abstract",
                "year": "year"
            }

        def __init__(self, papersData: list[PaperData]):
            self.search_df = pl.concat([el.search_df for el in papersData])
            self.citation_df = pl.concat([el.citation_df for el in papersData])
            self.reference_df = pl.concat([el.reference_df for el in papersData])
    return (JoinedPapers,)


@app.cell(hide_code=True)
def _(Path, hashlib, mo, re, requests):
    from crossref.restful import Works as cr_Works

    def clean_abstract(abstract: str|None) -> str:
        if abstract is not None:
            clean_text = re.sub(r'<[^>]+>', '', abstract)
            return clean_text.strip()
        else:
            return ""

    def try_get_abstract(doi: str):
        works = cr_Works()
        r = works.doi(doi)
        if r is None:
            return ""

        if "abstract" in r:
            return r.get("abstract", None)
        else:
            return ""

    def get_seminar_paper(keyword: str, best_n: int = 10):
        papers =[]

        url = f"http://api.semanticscholar.org/graph/v1/paper/search/bulk?query={keyword}&fields=title,abstract,citationCount,externalIds&sort=citationCount:desc"
        request = requests.get(url).json()
        with mo.status.spinner(subtitle=f"Searching seminar papers for {keyword}"):
            while len(papers) < 100:
                if "data" in request:
                    for paper in request["data"]:
                        papers.append(paper)

                if "token" not in request:
                    break

                request = requests.get(f"{url}&token={request['token']}").json()

        seminar_paper = [paper for paper in papers if "externalIds" in paper and "DOI" in paper["externalIds"]]
        seminar_paper = sorted(seminar_paper, key=lambda p: p.get("citationCount", 0), reverse=True)[:best_n]
        seminar_paper = [paper | {"DOI": paper["externalIds"]["DOI"]} for paper in seminar_paper]
        seminar_paper = [paper | {"abstract": clean_abstract(try_get_abstract(paper["DOI"]))} for paper in seminar_paper]
        return seminar_paper

    def get_seminar_text(keyword: str) -> list[str]:
        cache_path = Path(f"cache_seminar_papers/")
        cache_path.mkdir(parents=True, exist_ok=True)
        file_name = f"{hashlib.md5(keyword.encode("utf-8")).hexdigest()}.txt"
        if file_name in {f.parts[-1] for f in cache_path.iterdir()}:
            print(f"Using cache for seminar paper {keyword}: {file_name}")
            with open(cache_path.joinpath(file_name), "r") as file:
                seminar_papers = file.read().splitlines()
        else:
            seminar_papers = [f"{paper["title"]} {paper["abstract"]}" for paper in get_seminar_paper(keyword)]
            with open(cache_path.joinpath(file_name), "w+") as file:
                file.write("\n".join(seminar_papers))

        return seminar_papers
    return (
        clean_abstract,
        cr_Works,
        get_seminar_paper,
        get_seminar_text,
        try_get_abstract,
    )


@app.cell
def search_papers(JoinedPapers, SemanticScholarData):
    data = JoinedPapers([
        SemanticScholarData("\"batch activity\" + in + \"process modeling\""),
        SemanticScholarData("\"resource modeling\" + in + \"process simulation\""),
        SemanticScholarData("\"business process models\" \"manufacturing operations\""),
        SemanticScholarData("\"BPMN extensions\" \"manufacturing processes\""),
        SemanticScholarData("\"conceptual model\" + for + \"manufacturing simulation\""),
        SemanticScholarData("\"core manufacturing simulation data\"")
    ])
    # data = OpenAlexData(" AND ".join(words))
    return (data,)


@app.cell
def fetch_seminars_papers(get_seminar_text):
    # FIXME: don't change the value passed to these functions. The seminar paperes are not donwloaded anymore and a modified version of the cached one is beeing used
    bpmn_seminar_text = get_seminar_text("\"business process management\"")
    manufacturing_seminar_text = get_seminar_text("\"manufacturing environment\"")
    simulation_seminar_text = get_seminar_text("\"discrete event simulation\"")
    return (
        bpmn_seminar_text,
        manufacturing_seminar_text,
        simulation_seminar_text,
    )


@app.cell
def _(
    bpmn_seminar_text,
    manufacturing_seminar_text,
    sbert_model,
    simulation_seminar_text,
):
    seminars = {
        "BPMN": sbert_model.encode(bpmn_seminar_text),
        "manufacturing": sbert_model.encode(manufacturing_seminar_text),
        "simulation": sbert_model.encode(simulation_seminar_text)
    }
    return (seminars,)


@app.cell
def compute_similarity(data, mo, pl, sbert_model, seminars):
    all_dfs = pl.concat([data.filtered_related_papers_df, data.filtered_search_df]).unique("DOI")

    all_papers_list = (all_dfs
                       .with_columns(combined_text=pl.col("title") + " " + pl.col("abstract").fill_null(""))["combined_text"]
                       .to_list())

    papers_embedded = sbert_model.encode(all_papers_list, show_progress_bar=True)

    def sentence_similarity_spectre(field: str):
        seminar_texts = seminars[field]
        return [sbert_model.similarity(seminar_texts, paper).mean() for paper in mo.status.progress_bar(papers_embedded, title=f"Computing similarity for {field}", remove_on_exit=True)]

    def compute_similarity_with(field: str, algo) -> pl.Series:
        return pl.Series(name=f"{field}_similarity", values= algo(field))

    all_papers = all_dfs.with_columns([
        compute_similarity_with("BPMN", sentence_similarity_spectre),
        compute_similarity_with("manufacturing", sentence_similarity_spectre),
        compute_similarity_with("simulation", sentence_similarity_spectre),
    ]).unique("DOI")

    return (
        all_dfs,
        all_papers,
        all_papers_list,
        compute_similarity_with,
        papers_embedded,
        sentence_similarity_spectre,
    )


@app.cell
def _(all_papers):
    all_papers
    return


@app.cell
def _(all_papers, pl):
    import plotly.express as px

    plot_df = (all_papers
               .select("title", "BPMN_similarity", "manufacturing_similarity", "simulation_similarity")
               .with_columns(color_id=pl.col("title").cast(pl.Categorical).to_physical()))

    fig_parallel = px.parallel_coordinates(
            plot_df,
            dimensions=['BPMN_similarity', 'manufacturing_similarity', 'simulation_similarity'],
            color="color_id",
            title='Parallel Coordinates Plot of Paper Similarities',
            labels={
                'BPMN_similarity': 'BPMN',
                'manufacturing_similarity': 'Manufacturing',
                'simulation_similarity': 'Simulation'
            }
        )
    fig_parallel.show()
    return fig_parallel, plot_df, px


@app.cell
def _(mo, pl, sbert_model, seminars):
    s = pl.read_csv("/Users/marco/Downloads/export-data.csv")["Title"].to_list()
    d = sbert_model.encode(s)

    pl.from_dict({k: [sbert_model.similarity(seminar_texts, v).mean() for v in mo.status.progress_bar(d, title=f"Computing similarity for {k}")] for k, seminar_texts in seminars.items()}).with_columns(title=pl.Series(values=s)).with_columns(second_max=pl.col("BPMN")+pl.col("manufacturing")+pl.col("simulation")-pl.max_horizontal("BPMN", "manufacturing", "simulation")-pl.min_horizontal("BPMN", "manufacturing", "simulation"))
    return d, s


@app.cell(hide_code=True)
def _(mo):
    second_max = mo.ui.slider(0, 1, 0.01, show_value=True, label="Second max similarity threshold")
    min_citations = mo.ui.slider(0, 100, 1, show_value=True, label="Minimum number of citations")
    mo.vstack([second_max, min_citations])
    return min_citations, second_max


@app.cell
def _(all_papers, min_citations, pl, second_max):
    r = all_papers.with_columns(second_max=pl.col("BPMN_similarity")+pl.col("manufacturing_similarity")+pl.col("simulation_similarity")-pl.max_horizontal("BPMN_similarity", "manufacturing_similarity", "simulation_similarity")-pl.min_horizontal("BPMN_similarity", "manufacturing_similarity", "simulation_similarity")).filter((pl.col("second_max")>second_max.value) & (pl.col("citationCount")>min_citations.value))
    r
    return (r,)


@app.cell(hide_code=True)
def _(r):
    to_find = {
    "Workflow Resource Patterns: Identification, Representation and Tool Support",		
    "Using business process models for the specification of manufacturing operations",
    "[no semantic]A Proposal of BPMN Extensions for the Manufacturing Domain",
    "Proposal of BPMN extensions for modelling manufacturing processes",
    "A Visualization of Human Physical Risks in Manufacturing Processes Using BPMN",
    "Resource Modeling in Business Process Simulation",
    "The development of an ontology for describing the capabilities of manufacturing resources",
    "Ontology-Based Production Simulation with OntologySim",
    "Guiding principles for conceptual model creation in manufacturing simulation",
    "A conceptual framework for the generation of simulation models from process plans and resource configuration",
    "Core Manufacturing Simulation Data – a manufacturing simulation integration standard: overview and case studies",
    "Batch Activities in Process Modeling and Execution"
    }

    list(set(r["title"].to_list()).intersection(to_find))

    return (to_find,)


@app.cell(hide_code=True)
def _(r, to_find):
    list(to_find-set(r["title"].to_list()))
    return


if __name__ == "__main__":
    app.run()
