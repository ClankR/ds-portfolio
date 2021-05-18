##URL>Sentences UTILS

def get_content(url):
    import requests
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        page = requests.get(url, verify = False)
    return page

def get_sentences(page):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(page.content, 'html.parser')
    for script in soup(["script", "style"]):
        script.extract()    # rip it out
    title = soup.title.text
    import string
    import re
    vals = [x.strip().replace(u'\xa0', u' ') for x in soup.strings]
    sents = [re.sub(' +', ' ', x.strip()) for x in (" ".join(['\n' if val=='' else val for val in vals])).split('\n') if len(x)>20]
    # print(sents[:10])
    return sents, title