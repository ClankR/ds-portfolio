from loguru import logger
import click
import pandas as pd

# from creds import DOMKEY, DOMSECKEY
# CLIENT_ID = DOMKEY
# CLIENT_SECRET = DOMSECKEY

import warnings
warnings.filterwarnings("ignore")

import datetime
today = datetime.datetime.today()

@logger.catch
def previous_run(file_path = "../../DATA/RealEstate.csv"):
    tbl = pd.read_csv(file_path)
    from_date = tbl.date_run.max()
    return from_date

@logger.catch
def auth_run(client_id, client_secret):
    import requests
    response = requests.post(
            "https://auth.domain.com.au/v1/connect/token",
            data={
                "client_id": client_id,
                "client_secret": client_secret,
                "grant_type": "client_credentials",
                "scope": "api_listings_read api_addresslocators_read",
                "Content-Type": "text/json",
            },
            verify=False,
        )
    token = response.json()
    access_token = token["access_token"]
    auth = {"accept": "text/json", "Authorization": "Bearer " + access_token}
    return auth


@logger.catch
def get_data(from_date = "2019-12-24", auth = '', today = today, page_max = 5,
            suburb = 'Parramatta', state = 'NSW', postcode = 2150, 
            min_price = 650000, max_price = 1000000, 
            min_bed = 2, max_bed = 4, min_bath = 2, max_bath = 4,
            prop_type = ["House", "NewApartments", "apartmentUnitFlat"],
            surrounding = True):
    import json
    import requests
    import time
    import pandas as pd
    import warnings
    warnings.filterwarnings("ignore")
        
    base = pd.DataFrame()

    for i in range(1, page_max):
        # for j in range(1, res_max):
            # if (i % 2 == 0) & (j % 20 == 0):
            #     print(i, j)
            url = "https://api.domain.com.au/v1/listings/residential/_search"
            post_fields = {
                "listingType": "Sale",  # Rent
                "propertyType": prop_type,
                "minPrice": min_price,
                "maxPrice": max_price,
                "minBedrooms": min_bed,
                "maxBedrooms": max_bed,
                "minBathrooms": min_bath,
                "maxBathrooms": max_bath,
                "page": i,
                "pageSize": 100,#j  # 100,
                "locations": [
                    {
                        "state": state,
                        "region": "",
                        "area": "",
                        "suburb": suburb,
                        "postCode": postcode,
                        "includeSurroundingSuburbs": surrounding,
                    }
                ],
            "updatedSince": from_date,
            "sort": {
                "sortKey":"DateUpdated",
                "direction":"Descending"
            }
            }
            try:
                request = requests.post(url, headers=auth, json=post_fields, verify=False)
                if not request.ok:
                    logger.warning(request, request.content)
                    time.sleep(1)
                elif request.json() != [] and len(request.json())>0:
                    # print(len(request.json()))
                    for k in range(0,len(request.json())):
                        if request.json()[k].get("listing"):
                            data = request.json()[k].get("listing")
                            row = {
                                "id": data.get("id"),
                                "type": data.get("propertyDetails").get("propertyType"),
                                "price": data.get("priceDetails").get("displayPrice"),
                                "url": "https://www.domain.com.au/{}-{}-{}".format(
                                    data.get("propertyDetails")
                                    .get("displayableAddress")
                                    .replace(" ", "-")
                                    .replace("/", "-")
                                    .replace(",", "-")
                                    .lower(),
                                    data.get("propertyDetails").get("postcode"),
                                    data.get("id"),
                                ),
                                "url2": f"https://www.domain.com.au/{data.get('listingSlug')}",
                                "photo": data.get("media")[0].get("url"),
                                "lat": data.get("propertyDetails").get("latitude"),
                                "lon": data.get("propertyDetails").get("longitude"),
                                "add": data.get("propertyDetails").get("displayableAddress"),
                                "street": data.get("propertyDetails").get("street"),
                                "suburb": data.get("propertyDetails").get("suburb"),
                                "postcode": data.get("propertyDetails").get("postcode"),
                                "headline": data.get("headline"),
                                "desc": data.get("summaryDescription"),
                                "features": data.get("propertyDetails").get("features"),
                                "bath": data.get("propertyDetails").get("bathrooms"),
                                "bed": data.get("propertyDetails").get("bedrooms"),
                                "car": data.get("propertyDetails").get("carspaces"),
                                }
                            base = base.append(row, ignore_index=True)
                elif request.json().get('errors'):
                    print(request.json().get('errors'))
                elif len(request.json())==0:
                    ##Next page is empty so stop loop
                    break
            except:
                continue
    return base

def price(dump):
    import string
    import re
    translator = str.maketrans('', '', string.punctuation)
    pnc = str(dump.translate(translator))
    reg = re.findall(r'\b\d{3,7}\b', pnc)
    norm =  [int(x)*1000 if len(x)==3 else int(x) for x in reg if int(x)>500]
    if len(norm)==1:
        norm = (norm[0], norm[0])
    if len(norm)>1:
        norm = tuple(norm[0:2])
    if len(norm)==0:
        norm = (None, None)
    return norm

def table_prep(df):

    import datetime

    date = datetime.datetime.today().strftime("%Y-%m-%d")
    tbll = df.copy(deep = True) #.drop_duplicates() #can't drop dupes with a list column
    tbll['date'] = date
    logger.info(f"Rows: {len(tbll)}")

    tbll[['min_price', 'max_price']] = [price(x) for x in tbll.price]

    return tbll


def app_tbl(df, csv_name="../../DATA/RealEstate.csv"):
    import os

    # if file does not exist write header
    if not os.path.isfile(csv_name):
        df.to_csv(csv_name, index = False)
    else:  # else it exists so append without writing the header
        df.to_csv(csv_name, mode="a", header=False, index = False)


# @click.command()
# @click.option("--pm", default=2, help="Max Number of Pages")
# @click.option("--rm", default=2, help="Max Results per Page")
# def main(pm, rm):
#     if pm * rm > 500:
#         logger.warning("Combination of Page size and number exceeds free limit.")

#     redf = get_data(page_max=pm, res_max=rm)
#     if redf.empty:
#         logger.error('No Results')
#         return
#     tbll = table_prep(redf)
#     app_tbl(tbll)
#     logger.info("Table Build Complete")


# if __name__ == "__main__":
#     main()
