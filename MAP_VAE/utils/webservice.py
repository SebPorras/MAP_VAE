import urllib.request
import urllib.parse
import ssl
import requests, re
from requests.adapters import HTTPAdapter, Retry
from typing import List


""" This module is collection of functions for accessing the EBI REST web services,
    including sequence retrieval, searching, gene ontology, BLAST and ClustalW.
    The class EBI takes precautions taken as to not send too many requests when
    performing BLAST and ClustalW queries.

    See
    http://www.ebi.ac.uk/Tools/webservices/tutorials
    """

__ebiUrl__ = "http://www.ebi.ac.uk/Tools/"
__ebiGOUrl__ = "https://www.ebi.ac.uk/QuickGO/services/"
__uniprotUrl__ = "https://rest.uniprot.org/"
__ebiSearchUrl__ = "http://www.ebi.ac.uk/ebisearch/"
__NCBISearchUrl__ = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def fetch(entryId, dbName, format="fasta"):
    """
    Retrieve a single entry from a database
    entryId: ID for entry e.g. 'P63166' or 'SUMO1_MOUSE' (database dependent; examples for uniprotkb)
    dbName: name of database e.g. 'uniprotkb' or 'pdb' or 'refseqn'; see http://www.ebi.ac.uk/Tools/dbfetch/dbfetch/dbfetch.databases for available databases
    format: file format specific to database e.g. 'fasta' or 'uniprot' for uniprotkb (see http://www.ebi.ac.uk/Tools/dbfetch/dbfetch/dbfetch.databases)
    See http://www.ebi.ac.uk/Tools/dbfetch/syntax.jsp for more info re URL syntax

    """

    # Construct URL for uniprot
    if dbName.startswith("uniprot"):
        url = "{}{}/search?format={}&query={}".format(
            __uniprotUrl__, dbName, format, urllib.parse.quote(entryId)
        )

        # Get the entry
        try:

            data = requests.get(url).text
            if data.startswith("ERROR"):
                raise RuntimeError(data)
            return data

        except urllib.error.HTTPError as ex:
            raise RuntimeError(ex.read())

    # Construct URL
    if dbName.startswith("genbank"):

        url = "{}efetch.fcgi?db=nucleotide&id={}&rettype={}&retmode=text".format(
            __NCBISearchUrl__, entryId, format
        )

        # Get the entry
        try:
            context = ssl._create_unverified_context()
            data = urllib.request.urlopen(url, context=context).read().decode("utf-8")

            if data.startswith("ERROR"):
                raise RuntimeError(data)
            return data
        except urllib.error.HTTPError as ex:
            raise RuntimeError(ex.read())


def search(query, format="list", dbName="uniprot", limit=100, columns=""):
    """
    Retrieve multiple entries matching query from a database currently only via UniProtKB
    query: search term(s) e.g. 'organism:9606+AND+antigen'
    dbName: name of database e.g. 'uniprot', "refseq:protein", "refseq:pubmed"
    format: file format e.g. 'list', 'fasta' or 'txt'
    limit: max number of results (specify None for all results)
    See http://www.uniprot.org/faq/28 for more info re UniprotKB's URL syntax
    See http://www.ncbi.nlm.nih.gov/books/NBK25499/ for more on NCBI's E-utils
    """

    if dbName.startswith("uniprot"):
        # Construct URL
        if limit == None:  # no limit to number of results returned
            url = "{}{}/?format={}&query={}&columns={}".format(
                __uniprotUrl__, dbName, format, urllib.parse.quote(query), columns
            )

        # Get the entries

        try:
            data = urllib.request.urlopen(url).read().decode("utf-8")
            if format == "list":
                return data.splitlines()
            else:
                return data
        except urllib.error.HTTPError as ex:
            raise RuntimeError(ex.read().decode("utf-8"))

    elif dbName.startswith("refseq"):
        dbs = dbName.split(":")
        if len(dbs) > 1:
            dbName = dbs[1]

        base = "http://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

        url = (
            base + "esearch.fcgi?db={}&term={}+AND+srcdb_refseq["
            "prop]&retmax={}".format(dbName, urllib.parse.quote(query), str(limit))
        )

        # Get the entries
        try:
            data = urllib.request.urlopen(url).read().decode("utf-8")
            words = data.split("</Id>")
            words = [w[w.find("<Id>") + 4 :] for w in words[:-1]]
            if format == "list":
                return words
            elif format == "fasta" and len(words) > 0:
                url = base + "efetch.fcgi?db=" + dbName + "&rettype=fasta&id="
                for w in words:
                    url += w + ","
                data = urllib.request.urlopen(url).read().decode("utf-8")
                return data
            else:
                return ""
        except urllib.error.HTTPError as ex:
            raise RuntimeError(ex.read())
    return


def create_session():

    re_next_link = re.compile(r'<(.+)>; rel="next"')
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))

    return session, re_next_link


def get_next_link(headers, re_next_link):
    if "Link" in headers:
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)


def get_batch(batch_url, session, re_next_link):
    while batch_url:
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers, re_next_link)


def create_batch_url(
    ids: list,
    uniprot_cols: list,
    query_field: str = "accession",
    database: str = "uniprotkb",
):
    """Construct a URL for a REST API call to UniProt"""

    col_query = ",".join(uniprot_cols)

    batch_url = (
        f"https://rest.uniprot.org/{database}/search?"
        + "format=tsv&"
        + f"query={query_field}:("
        + " OR ".join(ids)
        + ")&fields="
        + col_query
        + "&size="
        + str(500)
    )

    return batch_url


def get_sequence_batch(
    ids: List[str],
    uniprotCols: List[str] = ["accession", "sequence"],
    query_field: str = "accession",
    database: str = "uniprotkb",
):

    # establish link to UniProt
    session, re_next_link = create_session()

    batch_url = create_batch_url(ids, uniprotCols, query_field, database)

    sequences = []
    progress = 0
    for batch, total in get_batch(batch_url, session, re_next_link):
        for line in batch.text.splitlines()[1:]:
            # will give a list of ['accession', 'sequence']
            data = line.split("\t")
            sequences.append(data)
            progress += 1

        print(f"{progress}/{total}")

    return sequences
