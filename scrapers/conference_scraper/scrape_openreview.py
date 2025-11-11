import requests
import pandas as pd
import time
import os
from datetime import datetime


class OpenReviewScraper:
    def __init__(
        self,
        conference="ICLR",
        start_year=2017,
        end_year=2024,
        fetch_decisions=False,
        fetch_scores=False,
    ):
        self.conference = conference
        self.start_year = start_year
        self.end_year = end_year
        self.fetch_decisions = fetch_decisions
        self.fetch_scores = fetch_scores
        self.papers = []

        self.api_v1 = "https://api.openreview.net/notes"
        self.api_v2 = "https://api2.openreview.net/notes"

        self.query_types = [
            "submission",
            "Submission",
            "Blind_Submission",
            "Withdrawn_Submission",
            "Rejected_Submission",
            "Desk_Rejected_Submission",
            "",
        ]

        self.request_delay = 1
        self.rate_limit_delay = 30

    def make_request(self, url):
        try:
            response = requests.get(url)
            data = response.json()

            if "name" in data and data["name"] == "RateLimitError":
                print(f"\nRate limited! Waiting {self.rate_limit_delay} seconds...")
                time.sleep(self.rate_limit_delay)
                response = requests.get(url)
                data = response.json()

            time.sleep(self.request_delay)
            return data

        except Exception as e:
            print(f"\nError making request to {url}: {e}")
            return None

    def build_url(self, year, query_type, offset=0):
        conf = self.conference

        if year <= 2017:
            if query_type == "":
                return None
            return f"{self.api_v1}?invitation={conf}.cc%2F{year}%2Fconference%2F-%2F{query_type}&offset={offset}"

        elif year <= 2023:
            if query_type == "":
                return None
            return f"{self.api_v1}?invitation={conf}.cc%2F{year}%2FConference%2F-%2F{query_type}&offset={offset}"

        else:
            query_suffix = f"/{query_type}" if query_type != "" else ""
            return f"{self.api_v2}?content.venueid={conf}.cc/{year}/Conference{query_suffix}&offset={offset}"

    def extract_paper_data(self, note, year, query_type):
        try:
            content = note.get("content", {})

            if year < 2024:
                title = content.get("title", "").strip()
                abstract = content.get("abstract", "").strip()
                keywords = content.get("keywords", [])
                authors_list = content.get("authors", [])

                if isinstance(authors_list, list):
                    authors = ", ".join(authors_list)
                else:
                    authors = authors_list if authors_list else ""

                if year == 2017:
                    if "authorids" in content:
                        author_ids = ", ".join(content["authorids"])
                    elif "author_emails" in content:
                        author_ids = content["author_emails"]
                    else:
                        author_ids = ""
                else:
                    author_ids = ", ".join(content.get("authorids", []))

            else:
                title = content.get("title", {}).get("value", "").strip()
                abstract = content.get("abstract", {}).get("value", "").strip()
                keywords = content.get("keywords", {}).get("value", [])

                if "authors" in content:
                    authors = ", ".join(content.get("authors", {}).get("value", []))
                    author_ids = ", ".join(
                        content.get("authorids", {}).get("value", [])
                    )
                else:
                    authors = ""
                    author_ids = ""

            if year <= 2020:
                author_ids = ""

            if "Withdrawn_Submission" in query_type:
                decision = "Withdrawn"
            elif "Desk_Rejected_Submission" in query_type:
                decision = "Desk rejected"
            elif "Rejected_Submission" in query_type:
                decision = "Reject"
            else:
                decision = ""

            paper = {
                "year": year,
                "id": note.get("forum", note.get("id", "")),
                "title": title,
                "abstract": abstract,
                "authors": authors,
                "author_ids": author_ids,
                "keywords": [k.lower() for k in keywords]
                if isinstance(keywords, list)
                else keywords,
                "decision": decision,
                "scores": [],
                "openreview_url": f"https://openreview.net/forum?id={note.get('forum', note.get('id', ''))}",
            }

            return paper

        except Exception as e:
            print(f"\nError extracting paper data: {e}")
            return None

    def fetch_decisions_and_scores_for_papers(self):
        print(f"Fetching decisions and scores (this will take several hours)")
        print(f"Total papers to process: {len(self.papers)}")

        for num, paper in enumerate(self.papers):
            if (num + 1) % 1000 == 0:
                print("*", end="", flush=True)
            elif (num + 1) % 100 == 0:
                print(".", end="", flush=True)

            year = paper["year"]
            forum_id = paper["id"]

            if not self.fetch_decisions and paper["decision"] != "":
                continue

            if year < 2024:
                forum_url = f"{self.api_v1}?forum={forum_id}"
            else:
                forum_url = f"{self.api_v2}?forum={forum_id}"

            data = self.make_request(forum_url)
            if data is None:
                continue

            notes = data.get("notes", [])

            if self.fetch_decisions and paper["decision"] == "":
                found_decision = False
                for note in notes:
                    content = note.get("content", {})

                    if "decision" in content:
                        decision = content["decision"]
                        if year >= 2024 and isinstance(decision, dict):
                            decision = decision.get("value", "")
                        paper["decision"] = decision
                        found_decision = True
                        break

                    if "withdrawal_confirmation" in content:
                        paper["decision"] = "Withdrawn"
                        found_decision = True
                        break

                    if "desk_reject_comments" in content:
                        paper["decision"] = "Desk rejected"
                        found_decision = True
                        break

                    if "recommendation" in content:
                        decision = content["recommendation"]
                        if isinstance(decision, dict):
                            decision = decision.get("value", "")
                        paper["decision"] = decision
                        found_decision = True
                        break

                    if "withdrawal" in content:
                        if content["withdrawal"] == "Confirmed":
                            paper["decision"] = "Withdrawn"
                            found_decision = True
                            break

            if self.fetch_scores:
                scores = []
                for note in notes:
                    content = note.get("content", {})

                    if "rating" in content:
                        rating = content["rating"]
                        if year >= 2024 and isinstance(rating, dict):
                            rating = rating.get("value", "")

                        if isinstance(rating, str):
                            try:
                                score = int(rating.split(":")[0])
                                scores.append(score)
                            except:
                                pass
                        elif isinstance(rating, (int, float)):
                            scores.append(int(rating))

                paper["scores"] = scores

        print()
        return self.papers

    def scrape_year(self, year):
        print(f"\nScraping {self.conference} {year}:")
        year_papers = []

        for query_type in self.query_types:
            offset = 0

            while True:
                url = self.build_url(year, query_type, offset)

                if url is None:
                    break

                data = self.make_request(url)

                if data is None:
                    break

                notes = data.get("notes", [])

                if len(notes) == 0:
                    break

                print(f"{len(notes)} ", end="", flush=True)

                for note in notes:
                    paper = self.extract_paper_data(note, year, query_type)
                    if paper:
                        year_papers.append(paper)

                if len(notes) < 1000:
                    break

                offset += 1000

        print(f"\nTotal papers for {year}: {len(year_papers)}")
        return year_papers

    def scrape_all(self):
        print(
            f"Scraping {self.conference} papers from {self.start_year} to {self.end_year}"
        )

        all_papers = []

        for year in range(self.start_year, self.end_year + 1):
            year_papers = self.scrape_year(year)
            all_papers.extend(year_papers)

        self.papers = all_papers
        print(f"Total papers scraped: {len(all_papers)}")

        if self.fetch_decisions or self.fetch_scores:
            self.fetch_decisions_and_scores_for_papers()

        return all_papers

    def filter_papers(self, min_abstract_length=100):
        print(
            f"\nFiltering papers with abstract length < {min_abstract_length} characters..."
        )

        filtered_papers = []
        removed_count = 0

        for paper in self.papers:
            if len(paper["abstract"]) >= min_abstract_length:
                filtered_papers.append(paper)
            else:
                removed_count += 1

        print(f"Removed {removed_count} papers with short abstracts")
        self.papers = filtered_papers

        return filtered_papers

    def to_dataframe(self):
        if not self.papers:
            print("No papers to convert!")
            return None

        df = pd.DataFrame(self.papers)

        df = df.sort_values(by=["year", "id"]).reset_index(drop=True)

        columns = [
            "year",
            "id",
            "title",
            "abstract",
            "authors",
            "author_ids",
            "decision",
            "scores",
            "keywords",
            "openreview_url",
        ]
        df = df[columns]

        return df

    def save_to_csv(self, output_file="papers.csv"):
        df = self.to_dataframe()
        if df is None:
            return None

        df_csv = df.copy()
        df_csv["scores"] = df_csv["scores"].apply(
            lambda x: ", ".join(map(str, x))
            if isinstance(x, list) and len(x) > 0
            else ""
        )
        df_csv["keywords"] = df_csv["keywords"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else x
        )

        text_columns = ["title", "abstract", "authors", "author_ids", "keywords"]
        for col in text_columns:
            if col in df_csv.columns:
                df_csv[col] = df_csv[col].apply(
                    lambda x: x.replace("\n", " ").replace("\r", " ")
                    if isinstance(x, str)
                    else x
                )

        df_csv.to_csv(
            output_file, index=False, encoding="utf-8", escapechar="\\", quoting=1
        )
        print(f"\nSaved {len(df_csv)} papers to {output_file}")

        return df

    def save_to_parquet(self, output_file="papers.parquet"):
        df = self.to_dataframe()
        if df is None:
            return None

        df.to_parquet(output_file)
        print(f"\nSaved {len(df)} papers to {output_file}")

        return df


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Scrape papers from OpenReview conferences"
    )
    parser.add_argument(
        "--conference",
        type=str,
        default="ICLR",
        help="Conference name (e.g., ICLR, NeurIPS)",
    )
    parser.add_argument("--start-year", type=int, default=2024, help="Start year")
    parser.add_argument("--end-year", type=int, default=2024, help="End year")
    parser.add_argument(
        "--fetch-decisions",
        action="store_true",
        help="Fetch accept/reject decisions (slow)",
    )
    parser.add_argument(
        "--fetch-scores",
        action="store_true",
        help="Fetch review scores (slow)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="csv",
        choices=["csv", "parquet"],
        help="Output format",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename (auto-generated if not provided)",
    )

    args = parser.parse_args()

    os.makedirs("results", exist_ok=True)

    print(
        f"Initializing scraper for {args.conference} ({args.start_year}-{args.end_year})"
    )
    if args.fetch_decisions or args.fetch_scores:
        print("WARNING: Fetching decisions/scores will take several hours!")

    scraper = OpenReviewScraper(
        conference=args.conference,
        start_year=args.start_year,
        end_year=args.end_year,
        fetch_decisions=args.fetch_decisions,
        fetch_scores=args.fetch_scores,
    )

    scraper.scrape_all()

    scraper.filter_papers(min_abstract_length=100)

    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        conf_lower = args.conference.lower()
        ext = "csv" if args.format == "csv" else "parquet"
        output_file = f"results/{conf_lower}_papers_{args.start_year}_{args.end_year}_{timestamp}.{ext}"
    else:
        output_file = args.output

    if args.format == "csv":
        df = scraper.save_to_csv(output_file)
    else:
        df = scraper.save_to_parquet(output_file)

    if df is not None:
        print("\nSummary Statistics:")
        print(df.groupby("year")["id"].count().rename("papers_per_year"))

        if args.fetch_decisions:
            print("\nDecision breakdown:")
            print(df["decision"].value_counts())

        if args.fetch_scores:
            # Calculate average scores
            scores_df = df[df["scores"].apply(lambda x: len(x) > 0)]
            if len(scores_df) > 0:
                avg_scores = scores_df["scores"].apply(lambda x: sum(x) / len(x))
                print(f"\nAverage score: {avg_scores.mean():.2f}")

        print("\n")
    else:
        print("\n" + "=" * 60)
        print("ERROR: No papers found!")
        print("=" * 60)
        print(f"The conference '{args.conference}' may not be hosted on OpenReview.")
        print("Conferences known to work: ICLR, NeurIPS, ICML, CoRL, TMLR")
        print("Conferences NOT on OpenReview: CVPR, ECCV, ICCV, ACL, EMNLP")


if __name__ == "__main__":
    main()
