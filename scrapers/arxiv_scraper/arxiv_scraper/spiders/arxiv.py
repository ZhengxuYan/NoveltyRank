import os
import re
import time
from datetime import datetime
import scrapy
from scrapy import Request
import urllib.parse
import csv
import xml.etree.ElementTree as ET


class ArxivSpider(scrapy.Spider):
    name = "arxiv"

    sleep_time = 3  # arXiv API recommends 3 seconds between requests
    articles_fetched = (
        0  # Counter for articles written to CSV (both preprints and accepted)
    )
    preprints_fetched = 0  # Counter for preprints written to main CSV
    accepted_fetched = 0  # Counter for accepted papers written to accepted CSV
    articles_processed = 0  # Counter for total articles processed (including skipped)
    start_index = 0  # For pagination
    current_category_index = 0  # Track which CS subcategory we're querying

    # Max results per API request - arXiv API allows up to 2000 per request
    max_results_per_request = 2000

    # Default date range for fetching papers (can be overridden with -a start_date=... -a end_date=...)
    start_date = "2025-04-01"
    end_date = "2025-06-30"

    # Selected Computer Science subcategories focused on ML/AI domains
    cs_categories = [
        "cs.LG",  # Machine Learning
        "cs.AI",  # Artificial Intelligence
        "cs.CV",  # Computer Vision
        "cs.CL",  # Computation and Language/NLP
        "cs.RO",  # Robotics
        "cs.CR",  # Cryptography and Security
    ]

    # Major ML/AI/CS conferences for acceptance detection
    conferences = [
        # Top-tier ML/AI
        "ICLR",
        "NeurIPS",
        "ICML",
        "AAAI",
        "IJCAI",
        "AISTATS",
        "UAI",
        "COLT",
        "ALT",
        # Computer Vision
        "CVPR",
        "ICCV",
        "ECCV",
        "WACV",
        "BMVC",
        "ACCV",
        "3DV",
        # NLP
        "ACL",
        "EMNLP",
        "NAACL",
        "COLING",
        "EACL",
        "AACL",
        "CoNLL",
        "SemEval",
        "COLM",
        # Robotics
        "ICRA",
        "IROS",
        "RSS",
        "CoRL",
        "CASE",
        "HUMANOIDS",
        # Data Mining & IR
        "KDD",
        "SIGIR",
        "WWW",
        "WSDM",
        "CIKM",
        "ICDM",
        "SDM",
        "ECIR",
        "RecSys",
        # Databases
        "SIGMOD",
        "VLDB",
        "ICDE",
        "EDBT",
        "CIDR",
        # Speech & Audio
        "ICASSP",
        "INTERSPEECH",
        "ASRU",
        "SLT",
        # Medical & Biology
        "MICCAI",
        "ISBI",
        "MIDL",
        "IPMI",
        # Multimedia
        "MM",
        "ICME",
        "ICIP",
        "ICPR",
        "ICDAR",
        # General AI
        "ECAI",
        "PRICAI",
        "FLAIRS",
        # Intelligent Systems
        "AAMAS",
        "ITSC",
        "IV",
        "ICAPS",
        "IJCNN",
        # Security & Privacy
        "CCS",
        "NDSS",
        "USENIX",
        "Oakland",
        "SP",
        # Learning Theory & Optimization
        "CLeaR",
        "ICLR",
        "OPT",
        # Fairness & Ethics
        "FAccT",
        "AIES",
        # Systems
        "SOSP",
        "OSDI",
        "NSDI",
        "EuroSys",
        "ATC",
    ]

    def __init__(self, *args, **kwargs):
        super(ArxivSpider, self).__init__(*args, **kwargs)

        # Create results directory if it doesn't exist
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # Open file using relative path
        results_file = os.path.join(results_dir, "arxiv_results_v2.csv")

        # Check if file exists and has content to determine if we need header
        file_exists = os.path.exists(results_file) and os.path.getsize(results_file) > 0

        # Open file in append mode
        self.file = open(results_file, "a", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)
        self.max_articles = 200000  # Maximum number of articles you want to fetch

        # Write the header row only if file is new/empty
        if not file_exists:
            self.writer.writerow(
                [
                    "arXiv ID",
                    "arXiv URL",
                    "PDF URL",
                    "DOI",
                    "Publication Date",
                    "Updated Date",
                    "Title",
                    "Authors",
                    "Author Affiliations",
                    "Abstract",
                    "Categories",
                    "Primary Category",
                    "Comment",
                    "Journal Reference",
                ]
            )

        # Open second file for accepted papers
        accepted_file = os.path.join(results_dir, "arxiv_results_accepted.csv")
        accepted_file_exists = (
            os.path.exists(accepted_file) and os.path.getsize(accepted_file) > 0
        )

        self.accepted_file = open(accepted_file, "a", newline="", encoding="utf-8")
        self.accepted_writer = csv.writer(self.accepted_file)

        # Write header for accepted papers file if needed
        if not accepted_file_exists:
            self.accepted_writer.writerow(
                [
                    "arXiv ID",
                    "arXiv URL",
                    "PDF URL",
                    "DOI",
                    "Publication Date",
                    "Updated Date",
                    "Title",
                    "Authors",
                    "Author Affiliations",
                    "Abstract",
                    "Categories",
                    "Primary Category",
                    "Comment",
                    "Journal Reference",
                ]
            )

    def close_spider(self, spider):
        self.file.close()
        self.accepted_file.close()

    def is_accepted_paper(self, comment, journal_ref):
        """
        Determine if a paper is accepted based on improved detection logic

        Args:
            comment: Paper comment field
            journal_ref: Journal reference field

        Returns:
            bool: True if paper appears to be accepted
        """
        if not comment and not journal_ref:
            return False

        # Skip workshops explicitly
        if re.search(r"\bworkshop\b", comment, re.IGNORECASE):
            return False

        # Check journal reference (strong signal)
        if journal_ref:
            return True

        comment_lower = comment.lower()

        # Pattern 1: Explicit acceptance phrases
        acceptance_patterns = [
            r"\baccepted\s+(at|to|by|for|in)\b",
            r"\bto\s+appear\s+(in|at)\b",
            r"\bto\s+be\s+published\b",
            r"\bpublished\s+(at|in|by|as)\b",  # Added "as" for "published as"
            r"\bappear(s|ing|ed)?\s+(at|in)\b",
            r"\bpresent(ed|ing)?\s+at\b",
        ]

        for pattern in acceptance_patterns:
            if re.search(pattern, comment_lower):
                return True

        # Pattern 2: Conference name + year (strong signal of acceptance)
        conf_pattern = "|".join(re.escape(conf) for conf in self.conferences)
        conf_year_pattern = rf"\b({conf_pattern})\s*[:\-]?\s*20[12][0-9]\b"
        if re.search(conf_year_pattern, comment, re.IGNORECASE):
            return True

        # Pattern 3: Conference + presentation type
        conf_context_pattern = (
            rf"\b({conf_pattern})\s+(camera|poster|oral|spotlight|paper)\b"
        )
        if re.search(conf_context_pattern, comment, re.IGNORECASE):
            return True

        # Pattern 4: "In [Conference]" or "At [Conference]"
        in_conf_pattern = rf"\b(in|at)\s+({conf_pattern})\b"
        if re.search(in_conf_pattern, comment, re.IGNORECASE):
            return True

        return False

    def build_api_url(self, start_index=0, category_index=None):
        """Build arXiv API query URL for specific CS subcategory in date range"""
        if category_index is None:
            category_index = self.current_category_index

        # Get current category
        if category_index >= len(self.cs_categories):
            return None  # No more categories to query

        category = self.cs_categories[category_index]

        # Format dates for arXiv API (YYYYMMDD)
        start_date_formatted = self.start_date.replace("-", "")
        end_date_formatted = self.end_date.replace("-", "")

        # Search specific CS subcategory with date range
        search_query = f"cat:{category} AND submittedDate:[{start_date_formatted} TO {end_date_formatted}]"

        query_params = {
            "search_query": search_query,
            "start": start_index,
            "max_results": self.max_results_per_request,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        api_url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
            query_params
        )
        return api_url

    def start_requests(self):
        self.logger.info(f"Fetching papers from {self.start_date} to {self.end_date}")
        self.logger.info("Using arXiv API (official method, no CAPTCHA restrictions)")
        self.logger.info(
            f"Searching {len(self.cs_categories)} Computer Science subcategories"
        )
        self.logger.info(f"Starting with category: {self.cs_categories[0]}")
        yield Request(
            url=self.build_api_url(start_index=0),
            callback=self.parse_api_response,
            meta={"dont_obey_robotstxt": True},
        )

    def parse_api_response(self, response):
        """Parse XML response from arXiv API"""
        self.logger.info(f"Response status: {response.status}")

        # Parse XML response
        try:
            root = ET.fromstring(response.body)
        except ET.ParseError as e:
            self.logger.error(f"Failed to parse XML: {e}")
            return

        # Define namespaces
        namespaces = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        # Get total results
        total_results_elem = root.find("atom:totalResults", namespaces)
        total_results = (
            int(total_results_elem.text) if total_results_elem is not None else 0
        )
        self.logger.info(f"Total results available: {total_results}")

        # Get all entry elements (papers)
        entries = root.findall("atom:entry", namespaces)
        self.logger.info(
            f"Found {len(entries)} papers in this batch (start_index: {self.start_index})"
        )

        # Process each paper (if any)
        for entry in entries:
            self.articles_processed += 1

            try:
                # Extract paper details from XML
                title_elem = entry.find("atom:title", namespaces)
                title = (
                    title_elem.text.strip().replace("\n", " ")
                    if title_elem is not None
                    else ""
                )

                # Get arXiv URL and ID from the id element
                id_elem = entry.find("atom:id", namespaces)
                arxiv_url = id_elem.text if id_elem is not None else ""
                # Extract arXiv ID (e.g., "2501.12345" from "http://arxiv.org/abs/2501.12345")
                arxiv_id = arxiv_url.split("/abs/")[-1] if arxiv_url else ""

                # Get PDF URL from links
                pdf_url = ""
                link_elems = entry.findall("atom:link", namespaces)
                for link in link_elems:
                    if link.get("title") == "pdf":
                        pdf_url = link.get("href", "")
                        break

                # Get DOI
                doi_elem = entry.find("arxiv:doi", namespaces)
                doi = doi_elem.text if doi_elem is not None else ""

                # Get published date
                published_elem = entry.find("atom:published", namespaces)
                if published_elem is not None:
                    published_date = datetime.fromisoformat(
                        published_elem.text.replace("Z", "+00:00")
                    )
                    publication_date = published_date.date().isoformat()
                else:
                    publication_date = ""

                # Get updated date
                updated_elem = entry.find("atom:updated", namespaces)
                if updated_elem is not None:
                    updated_date = datetime.fromisoformat(
                        updated_elem.text.replace("Z", "+00:00")
                    )
                    updated_date_str = updated_date.date().isoformat()
                else:
                    updated_date_str = ""

                # Filter by date range - skip papers outside our range
                if publication_date:
                    if publication_date < self.start_date:
                        # We've gone past our date range (papers are sorted descending)
                        # Stop processing further
                        self.logger.info(
                            f"Reached papers before start_date ({self.start_date}). Stopping."
                        )
                        self.logger.info(
                            f"Processed {self.articles_processed} papers total, "
                            f"fetched {self.articles_fetched} within date range "
                            f"(Preprints: {self.preprints_fetched}, Accepted: {self.accepted_fetched})"
                        )
                        return
                    elif publication_date > self.end_date:
                        # Skip papers that are too recent
                        self.logger.debug(
                            f"Skipping paper from {publication_date} (after end_date)"
                        )
                        continue

                # Get authors and their affiliations
                author_elems = entry.findall("atom:author", namespaces)
                authors = []
                affiliations = []
                for author in author_elems:
                    name_elem = author.find("atom:name", namespaces)
                    if name_elem is not None:
                        authors.append(name_elem.text)

                    # Get affiliation if available
                    affiliation_elem = author.find("arxiv:affiliation", namespaces)
                    if affiliation_elem is not None:
                        affiliations.append(
                            f"{name_elem.text}: {affiliation_elem.text}"
                        )

                authors_str = "; ".join(authors)
                affiliations_str = "; ".join(affiliations) if affiliations else ""

                # Get abstract
                summary_elem = entry.find("atom:summary", namespaces)
                abstract = (
                    summary_elem.text.strip().replace("\n", " ")
                    if summary_elem is not None
                    else ""
                )

                # Get categories
                category_elems = entry.findall("atom:category", namespaces)
                categories = []
                for category in category_elems:
                    term = category.get("term")
                    if term:
                        categories.append(term)
                categories_str = "; ".join(categories)

                # Get primary category
                primary_category_elem = entry.find("arxiv:primary_category", namespaces)
                primary_category = (
                    primary_category_elem.get("term")
                    if primary_category_elem is not None
                    else ""
                )

                # Filter: only keep papers where primary category is one of our target categories
                if primary_category not in self.cs_categories:
                    self.logger.debug(
                        f"Skipping paper '{title[:50]}...' - primary category '{primary_category}' not in target categories"
                    )
                    continue

                # Get comment (often contains page count, conference info, etc.)
                comment_elem = entry.find("arxiv:comment", namespaces)
                comment = (
                    comment_elem.text.strip().replace("\n", " ")
                    if comment_elem is not None
                    else ""
                )

                # Get journal reference
                journal_ref_elem = entry.find("arxiv:journal_ref", namespaces)
                journal_ref = (
                    journal_ref_elem.text.strip().replace("\n", " ")
                    if journal_ref_elem is not None
                    else ""
                )

                # Use improved acceptance detection
                is_accepted = self.is_accepted_paper(comment, journal_ref)

                # Skip workshops
                if comment and "workshop" in comment.lower():
                    self.logger.debug(
                        f"Skipping paper '{title[:50]}...' - comment contains 'workshop'"
                    )
                    continue

                # Determine which file to write to
                if is_accepted:
                    # Write accepted papers to separate file
                    self.accepted_writer.writerow(
                        [
                            arxiv_id,
                            arxiv_url,
                            pdf_url,
                            doi,
                            publication_date,
                            updated_date_str,
                            title,
                            authors_str,
                            affiliations_str,
                            abstract,
                            categories_str,
                            primary_category,
                            comment,
                            journal_ref,
                        ]
                    )
                    self.accepted_fetched += 1
                    self.logger.debug(
                        f"Saved accepted paper '{title[:50]}...' to accepted file"
                    )
                else:
                    # Write preprints to main file
                    self.writer.writerow(
                        [
                            arxiv_id,
                            arxiv_url,
                            pdf_url,
                            doi,
                            publication_date,
                            updated_date_str,
                            title,
                            authors_str,
                            affiliations_str,
                            abstract,
                            categories_str,
                            primary_category,
                            comment,
                            journal_ref,
                        ]
                    )
                    self.preprints_fetched += 1

                self.articles_fetched += 1

                # Check if we've reached the limit
                if self.articles_fetched >= self.max_articles:
                    self.logger.info(
                        f"Reached the limit of {self.max_articles} articles."
                    )
                    return

            except Exception as e:
                self.logger.error(f"Error processing entry: {e}")
                continue

        # Update start index for next batch
        self.start_index += len(entries)

        self.logger.info(
            f"Total articles fetched so far: {self.articles_fetched} "
            f"(Preprints: {self.preprints_fetched}, Accepted: {self.accepted_fetched})"
        )

        # Check if there are more results to fetch
        if self.articles_fetched >= self.max_articles:
            # Already reached our limit
            self.logger.info(
                f"Reached the limit of {self.max_articles} articles. Stopping."
            )
            return

        if len(entries) > 0:
            # Got entries, keep going with same category
            self.logger.info(
                f"Fetching next batch from {self.cs_categories[self.current_category_index]} (start_index: {self.start_index})"
            )
            self.logger.info(
                f"Sleeping for {self.sleep_time} seconds (arXiv API rate limiting)"
            )
            time.sleep(self.sleep_time)

            yield Request(
                url=self.build_api_url(start_index=self.start_index),
                callback=self.parse_api_response,
                meta=response.meta,
            )
        else:
            # No entries - move to next category
            self.current_category_index += 1
            self.start_index = 0  # Reset start index for new category

            if self.current_category_index >= len(self.cs_categories):
                # All categories exhausted
                self.logger.info(
                    f"Finished all {len(self.cs_categories)} categories! "
                    f"Total articles fetched: {self.articles_fetched} "
                    f"(Preprints: {self.preprints_fetched}, Accepted: {self.accepted_fetched})"
                )
                return

            # Move to next category
            next_category = self.cs_categories[self.current_category_index]
            self.logger.info(f"Finished category, moving to next: {next_category}")
            self.logger.info(
                f"Sleeping for {self.sleep_time} seconds (arXiv API rate limiting)"
            )
            time.sleep(self.sleep_time)

            yield Request(
                url=self.build_api_url(start_index=0),
                callback=self.parse_api_response,
                meta=response.meta,
            )
