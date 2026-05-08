from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_td = False
        self.row = []
    def handle_starttag(self, tag, attrs):
        if tag == 'table': print('-'*80)
        elif tag == 'tr': self.row = []
        elif tag in ('td', 'th'): self.in_td = True
    def handle_endtag(self, tag):
        if tag == 'tr': print(' | '.join(self.row))
        elif tag in ('td', 'th'): self.in_td = False
    def handle_data(self, data):
        data = data.strip()
        if self.in_td and data: self.row.append(data)

with open('results/paper_results.html', 'r', encoding='utf-8') as f:
    MyHTMLParser().feed(f.read())
