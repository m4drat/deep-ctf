import time
import requests

from alive_progress import alive_bar
from bs4 import BeautifulSoup


BASE_URL = 'https://ctftime.org/stats/2021?page={}'
TOTAL_PAGES = 738


def main():
    with open('teams.txt', 'a+') as output_file:
        with alive_bar(TOTAL_PAGES - 1) as bar:
            for page in range(1, TOTAL_PAGES):
                url = BASE_URL.format(page)
                response = requests.get(
                    url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', {'class': 'table table-striped'})
                rows = table.find_all('tr')
                for row in rows:
                    # Find the team name in the table row
                    team_url = row.find(
                        'a', href=lambda href: href and 'team' in href)
                    if team_url:
                        team_name = team_url.text
                        # Skip invalid team names
                        if team_name.isascii() and 'http' not in team_name:
                            output_file.write(team_name + '\n')

                if page % 10 == 0:
                    output_file.flush()

                time.sleep(2)
                bar()


if __name__ == '__main__':
    main()
