import os
import tempfile

import requests
from bs4 import BeautifulSoup

documents_directory = None
test_size = None
# Check if TEST environment variable is set to true
if os.getenv("TEST", "false").lower() == "true":
    print("Running in test mode")
    test_size = 1
    # Use a temporary directory
    documents_directory = tempfile.mkdtemp()
    # Fetch a random wiki page content
    wiki_random_url = "https://en.wikipedia.org/wiki/Special:Random"
    response = requests.get(wiki_random_url)
    soup = BeautifulSoup(response.content, "html.parser")
    wiki_content = soup.get_text()
    # Write the content to a txt file in the temporary directory
    with open(os.path.join(documents_directory, "random_wiki_page.txt"), "w") as file:
        file.write(wiki_content)
else:
    documents_directory = os.getenv("DOCS_DIR", "./docs")
    test_size = 10
