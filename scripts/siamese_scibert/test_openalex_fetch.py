import sys
import os

# Add the current directory to sys.path so we can import openalex_utils
sys.path.append(os.path.join(os.path.dirname(__file__)))

from openalex_utils import OpenAlexFetcher, fetch_comprehensive_author_info

def test_fetch_author():
    fetcher = OpenAlexFetcher()
    # Yann LeCun's OpenAlex ID or just search by name
    author_name = "Yann LeCun"
    print(f"Fetching info for {author_name}...")
    
    info = fetch_comprehensive_author_info(author_name, fetcher, debug=True)
    
    if info:
        print("\n--- Fetched Info ---")
        print(f"Name: {info.get('name')}")
        print(f"Affiliations: {info.get('affiliations')}")
        
        if "affiliations" in info:
            print("\nSUCCESS: 'affiliations' field is present.")
            affs = info["affiliations"]
            if "New York University" in affs and "AT&T" not in affs:
                 print("SUCCESS: Affiliations seem correctly filtered (NYU present, AT&T absent).")
            else:
                 print("WARNING: Check affiliation filtering logic.")
        else:
            print("\nFAILURE: 'affiliations' field is MISSING.")
    else:
        print("\nFAILURE: Could not fetch author info.")

if __name__ == "__main__":
    test_fetch_author()
