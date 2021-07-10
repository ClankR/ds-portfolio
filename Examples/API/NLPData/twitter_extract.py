from creds import TWITKEY, TWITSECKEY, TWITACC, TWITTOK
import tweepy as tw
import pandas as pd

class twit_api:
    '''
    The below class is an example of instantiating a connection to the twitter api (using free developer credentials you will need to save in a file called creds.py).
    Using this connection, we can then pull a week's worth of tweets for our chosen search term.
    '''
    def __init__(self, TWITKEY = TWITKEY, TWITSECKEY = TWITSECKEY, 
                TWITACC = TWITACC, TWITTOK = TWITTOK):
        "Following Tweepy documentation found at https://www.tweepy.org/."
        auth = tw.OAuthHandler(TWITKEY, TWITSECKEY)
        auth.set_access_token(TWITACC, TWITTOK)
        api = tw.API(auth, wait_on_rate_limit=True)

        self.api = api

    def scrape(self, search_words = '', url_check = False):
        "Basic checks to dump out a valuable and ready-to-use dataframe of Tweets data and metadata."
        search = f'{search_words} -filter:retweets'
        tweets = tw.Cursor(self.api.search,
                q=search,
                lang="en",
    #               since=date_since, 
    #               before="2019-01-03",
                tweet_mode="extended").items(10000)

        data = [[tweet.user.name, tweet.user.location,
                tweet.user.profile_background_image_url_https,
                tweet.user.followers_count,
                
                tweet.created_at, 

                tweet.full_text, 
                
                tweet._json.get('retweet_count',0), 
                tweet.favorite_count,#_json.get('favourite_count', 0),

                [hsh.get('url') for hsh in tweet.entities.get('urls', {})],
                [hsh.get('text') for hsh in tweet.entities.get('hashtags', {})]
            ] for tweet in tweets]

        tt = pd.DataFrame(data=data, 
                        columns=['name', 'location', 'picurl', 'followers',
                            'dt', 'text', 'rts', 'favs', 'urls', 'hashtags'])
        
        print(f'Number of tweets scraped: {len(tt)}')
        
        import re
        tt['cleantext'] = [re.sub('|'.join(row.urls), '', row.text) for i, row in tt.iterrows()]
        
        if url_check:
            real_u = []
            import warnings
            warnings.simplefilter("ignore")
            import requests

            for i, url in enumerate(tt.urls):
                if i%1000==0:
                    print(i)
            #     print(url)
                try:
                    if len(url)>0:
                        r = requests.get(url[-1], verify = False).url
                        real_u.append(r)
                    else:
                        real_u.append('nourl')
                except:
                    real_u.append('nourl')

            tt['propurl'] = real_u
        else:
            tt['propurl'] = tt.urls

        self.extract = tt
