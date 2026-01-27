import os
import re

import requests


class QRUtils:
    def __init__(self):
        pass

    @staticmethod
    def read_covid_narratives(output_folder, url, num_docs=115):
        os.makedirs(output_folder, exist_ok=True)
        for doc_count in range(1, num_docs + 1): # 1 to 115 inclusive, as per available documents
            _url = f"https://{url}/items/show/{doc_count}"
            html = requests.get(_url).text
            # Extract <a class="download-file" href
            pattern = r'<a class="download-file" href="(.*?)">'
            # find first match
            match = re.search(pattern, html)
            if match:
                # Extract the URL
                file_url = match.group(1)
                # sanitize the URL
                file_url = file_url.replace("&amp;", "&")
                print(f"Downloading file from {file_url}")
                # if doc_{doc_count}.pdf") exists in output_folder, skip download
                if os.path.exists(os.path.join(output_folder, f"doc_{doc_count}.pdf")):
                    print(f"File doc_{doc_count}.pdf already exists, skipping download")
                    continue
                # Download the file
                response = requests.get(file_url)
                # Save the file to the output folder
                with open(
                    os.path.join(output_folder, f"doc_{doc_count}.pdf"), "wb"
                ) as f:
                    f.write(response.content)
            else:
                print(f"No match found for document {doc_count}")


    @staticmethod
    def print_table(table):
        col_width = [max(len(x) for x in col) for col in zip(*table)]
        for line in table:
            print(
                "| "
                + " | ".join(
                    "{:{}}".format(x, col_width[i]) for i, x in enumerate(line)
                )
                + " |"
            )


if __name__ == "__main__":
    # Example usage
    qr_utils = QRUtils()
    qr_utils.read_covid_narratives("/tmp/covid_narratives")
