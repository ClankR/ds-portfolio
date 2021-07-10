# import folium



def popup_func(row, string_var = [], photo_col = None):
    import folium
    from folium import IFrame
    import base64
    from io import BytesIO
    from PIL import Image
    import numpy as np
    import requests
    
    """Add row column names to include that information in popup bubbles."""
    st = ""
    for item in string_var:
        st = st+str(item.capitalize())+": "+str(row[item])+"<br>"
    
    from PIL import Image
    # import requests
    from io import BytesIO

    import warnings 
    warnings.filterwarnings('ignore')
    response = requests.get(row[photo_col], verify = False)
    img = Image.open(BytesIO(response.content))
#     img    
#     image_object = work_bucket.Object(row.img)
#     in_mem =    io.BytesIO(image_object.get()['Body'].read())
#     img = IM.open(in_mem)
    h,w = img.size
        
    img.thumbnail((np.round(h/5), np.round(350)))
        #1, 350) #np.round((350/w)*h)
        #(np.round(h/5), np.round(w/5)))
    imgByteArr = BytesIO()
    img.save(imgByteArr, format='PNG', optimize=True,quality=10)
    
    
    img_str = base64.b64encode(imgByteArr.getvalue()).decode()
    #base64.b64encode(open("{}".format(row.file), 'rb').read())
    pop = folium.Popup(IFrame(
                '<div style="font-family: Arial">{}<br><img src="data:image/;base64,{}">'.format(str(st),
                                                                                                
                                                                                          img_str)+
                '</div>',
                                              width = 350, height = 350

                                              ), 
                                                       max_width=2650)
    return(pop)


def icon_func(map_df, group_col, sort_col = None, colour_shift = 0):
    """
    Returns icons, icon colours, and an appropriate legend.
    """
    if(colour_shift>18):
        print("To big of colour spectrum shift")
        return
    
    fa = ['random', 'cog', 'magnet', 'pencil', 'wrench', 'signal',  
          'tag', 'certificate', 'volume-off',  'th-large', 
          'map-marker', 'plus', 
         'file', 'refresh', 'align-center', 'link', 
     'repeat', 'th', 'heart', 'briefcase']
    
    cols = ['red', 'blue', 'green', 'purple', 'orange', 'darkred',
                 'lightred', 'beige', 'darkblue', 'darkgreen', 'cadetblue',
                 'darkpurple', 'pink', 'lightblue', 'lightgreen',
                 'gray', 'black', 'lightgray']*2
    if sort_col:
        ics = map_df.sort_values(sort_col)[group_col].unique()
    else:
        ics = map_df[group_col].unique()
    
    if(len(ics)>len(fa)):
        print("Too many groups")
        return
    
    
    icns = dict(zip(ics,
                fa[:len(ics)]))
    clrs = dict(zip(ics, 
                cols[colour_shift:(colour_shift+len(ics))]))
    
    leg = ""
    for key, item in icns.items():
        leg = leg + """{}&nbsp; <i class="fa fa-{} fa-2x"
                       style="color:{}"></i><br><br>""".format(key, item, clrs[key])
    legh = len(icns.items())*56
    
    legend_html = """
         <div style="position: fixed; 
         bottom: 25px; left: 25px; width: 120px; height: {}px; text-align:center;
         border:2px solid grey; background-color: white; z-index:9999; font-size:14px;
         "><h4 style = "text-align:center">Legend</h4>
         {}
          </div>
         """.format(legh, leg[:-8])
    
    return(icns, clrs, legend_html)


def create_map(map_df, lat_col, lon_col, group_col, strings = [], colour_shift = 0,
        zoom = 14, height = 900, disable_cluster = None, all = False,
        save_map_name = None, sort_col = None, photo_col = None):
    
    import folium
    from folium.plugins import MarkerCluster

    if disable_cluster is None:
            disable_cluster = zoom+2

    focus = map_df.groupby(lambda _ : True).agg({lat_col:'mean', lon_col:'mean'}).iloc[0]
    lat,lon = focus[lat_col], focus[lon_col] 
    f = folium.Figure(height=height)
    m = folium.Map(location=[lat, lon], zoom_start = zoom)#,  max_zoom=max_zoom)
    marker_cluster = MarkerCluster(options = {'disableClusteringAtZoom':disable_cluster})

    icns, clrs, legend_html = icon_func(map_df=map_df, 
                                        group_col = group_col, sort_col = sort_col,
                                        colour_shift = colour_shift)
    
    map_df.apply(lambda row:folium.Marker(location=[row[lat_col], row[lon_col]],
                                            popup = popup_func(row, strings, photo_col),
                                            icon = folium.Icon(color=clrs[row[group_col]], 
                                                               icon_color = "white",
                                                               prefix = 'fa',
                                                               icon = icns[row[group_col]])
                                               ).add_to(marker_cluster), axis=1) 
    
    # h = folium.Marker(location=[-33.81168,151.0113152],
    #                   icon = folium.Icon(color = 'red'))
    # m.add_child(h)

    m.add_child(marker_cluster)
    f.add_child(m)
    f.get_root().html.add_child(folium.Element(legend_html))
    if save_map_name:
        f.save(save_map_name)
    if all:
        return f,m
    else:
        return f

# ##Build Map    
# def map(map_df, 
#             lat_col = 'LATITUDE', lon_col = 'LONGITUDE', 
#             group_col = 'PLEASE SELECT', #e.g. 'PRIMARY_TECHNOLOGY'
#             string_var = [], #e.g. ['LOCATION_ID', 'PRIMARY_TECHNOLOGY']
            
#             height = 900,
#             zoom = 10,
#             colour_shift = 0,
#             disable_cluster = None,
#             max_zoom = 18,
            
#             save_map_name = "",
#             disable_legend = False,
#             geocode = False,
            
#             icon_func = icons, 
#             popup_func = popup
#             ):
    
#     import folium
# #     from folium.plugins import MarkerCluster
    
    
#     height = height
#     zoom = zoom
#     if disable_cluster is None:
#         disable_cluster = zoom+2
    
    
#     focus = map_df.groupby(lambda _ : True).agg({lat_col:'mean', lon_col:'mean'}).iloc[0]
#     lat,lon = focus[lat_col], focus[lon_col] 
    
#     icns, clrs, legend_html = icon_func(map_df=map_df, 
#                                         group_col = group_col, 
#                                         colour_shift = colour_shift)
    
   
#     f = folium.Figure(height=height)
#     m = folium.Map(location=[lat, lon], zoom_start = zoom,  max_zoom=max_zoom)
#     marker_cluster = MarkerCluster(options = {'disableClusteringAtZoom':disable_cluster}) 

    
#     if(geocode==True):
    
#         from geopy.geocoders import Nominatim
#         from geopy.exc import GeocoderTimedOut
#         geolocator = Nominatim(user_agent="hmm")
    
#         def do_geocode(lat, lon):
#             try:
#                 return geolocator.reverse("{}, {}".format(lat, lon), timeout=600)
#             except GeocoderTimedOut:
#                 return geolocator.reverse("{}, {}".format(lat, lon), timeout=600)
    
#         if(len(map_df)>500):
#             print('- Geocoding may take some time as there are over 500 addresses -\n')
#         print("- Reverse Geocoding -")
#         map_df.loc[:,('GEOCODE')] = map_df.apply(lambda row:("<br>"+do_geocode(row[lat_col], row[lon_col]).address), axis = 1)
#         string_var = string_var+['GEOCODE']
        
    
    
#     map_df.apply(lambda row:folium.Marker(location=[row[lat_col], row[lon_col]],
#                                             popup = popup_func(row, string_var),
#                                             icon = folium.Icon(color=clrs[row[group_col]], 
#                                                                icon_color = "white",
#                                                                prefix = 'fa',
#                                                                icon = icns[row[group_col]])
#                                                ).add_to(marker_cluster), axis=1) 


#     m.add_child(marker_cluster)
#     f.add_child(m)
    
#     if(disable_legend == False):
#         f.get_root().html.add_child(folium.Element(legend_html))
    
#     if(save_map_name != ""):
#         f.save("{}.html".format(save_map_name))
    
#     return(f)