import json
import requests
import time
import asyncio
import aiohttp
import gc
import hashlib
from concurrent.futures import ThreadPoolExecutor
from shapely.geometry import shape, Polygon, MultiPolygon, Point, MultiPoint, mapping, point
import os
import xml.etree.ElementTree as ET
from qwikidata.entity import WikidataItem, WikidataLexeme, WikidataProperty
from qwikidata.linked_data_interface import get_entity_dict_from_api
from operator import itemgetter
# import ctypes
# libc = ctypes.CDLL("libc.so.6")
from urllib.parse import urlparse

def get_polygon_for_city_osmid(osmID):
  status_code = 0
  while status_code != 200:
    url = "https://nominatim.openstreetmap.org/details.php?osmtype=R&osmid=" + osmID + "&class=boundary&addressdetails=1&hierarchy=0&group_hierarchy=1&format=json&polygon_geojson=1"
    try:
      resp_text, status_code = get_request_cached(url)
      if status_code == 200:
        jsonResult = json.loads(resp_text)
        polygon = jsonResult['geometry']
        polygon_shape = shape(polygon)
        return polygon_shape
      else:
        del resp_text
        time.sleep(2)
    except Exception as e:
      print(e)
      time.sleep(2)

def get_raw_overpass_for_city_block(osmId, polygon, block_index):
  if os.path.exists(f"./osm_cache/{osmId}_{block_index + 1}"):
    file = open(f"./osm_cache/{osmId}_{block_index + 1}")
    file_content = file.read()
    file.close()
    return file_content

  polygon_all_x = []
  polygon_all_y = []
  if type(polygon) is Polygon:
    polygon_all_y = polygon.exterior.xy[0]
    polygon_all_x = polygon.exterior.xy[1]
  elif type(polygon) is MultiPolygon:
    for geom in polygon.geoms:
      polygon_all_y.extend(geom.exterior.xy[0])
      polygon_all_x.extend(geom.exterior.xy[1])
  min_x = min(polygon_all_x)
  min_y = min(polygon_all_y)
  max_x = max(polygon_all_x)
  max_y = max(polygon_all_y)
  x_iter = (max_x - min_x) / 3
  y_iter = (max_y - min_y) / 3
  #print(f"min_x {min_x}, min_y {min_y}, max_x {max_x}, max_y {max_y}, x_iter {x_iter}, y_iter {y_iter}")

  i = 0
  block_found = False
  for x_i in range(0, 3):
    for y_i in range(0, 3):
      small_min_x = min_x + x_i * x_iter
      small_min_y = min_y + y_i * y_iter
      small_max_x = min_x + (x_i + 1) * x_iter
      small_max_y = min_y + (y_i + 1) * y_iter
      i += 1
      if i == block_index:
        block_found = True
        break
    if block_found:
      break

  status_code = None
  while status_code != 200:
    try:
      url = f"https://overpass-api.de/api/map?bbox={small_min_y},{small_min_x},{small_max_y},{small_max_x}"
      resp_text, status_code = get_request_cached(url)
      if status_code == 200:
        r_body = resp_text
      else:
        del resp_text
        time.sleep(5)
    except:
      None
  return r_body


def process_overpass_document(overpass_raw, cityName, city_polygon_shape):
  lines = []

  tree = ET.ElementTree(ET.fromstring(overpass_raw))
  root = tree.getroot()
  nnn = []
  for node in root.iter('node'):
    nnn.append(node)
  for node in root.iter('way'):
    nnn.append(node)

  for node in nnn:
    if 'lat' not in node.attrib or 'lon' not in node.attrib:
      continue

    # check if has wikidata value
    wikidata_value = None
    for tag in node.iter('tag'):
      if tag.attrib['k'] == 'wikidata':
        wikidata_value = tag.attrib['v']

    if wikidata_value is None:
      continue

    # check if object coordinates is in city
    lat = float(node.attrib['lat'])
    lng = float(node.attrib['lon'])
    if not city_polygon_shape.contains(Point(lng, lat)):
      continue

    # parse object type
    is_shop = False
    is_tourism = False
    is_leisure = False
    for tag in node.iter('tag'):
      if tag.attrib['k'] == 'shop':
        is_shop = True
      if tag.attrib['k'] == 'tourism':
        is_tourism = True
      if tag.attrib['k'] == 'leisure':
        is_leisure = True
    is_other = not is_shop and not is_tourism and not is_leisure

    # parse qwikidata
    retry_count = 0
    qwikidata_loaded = False
    while True:
      try:
        if (os.path.exists(f"./cache/{wikidata_value}")):
          file = open(f"./cache/{wikidata_value}", "r")
          qwikidata_dict = json.loads(file.read())
          file.close()
        else:
          qwikidata_dict = get_entity_dict_from_api(wikidata_value)
          file = open(f"./cache/{wikidata_value}", "w")
          file.write(json.dumps(qwikidata_dict))
          file.close()
        qwikidata_loaded = True
        break
      except:
        time.sleep(0.5)
        retry_count += 1
        if retry_count == 3:
          break
    if qwikidata_loaded is False:
      continue
    claims = qwikidata_dict["claims"]

    # parse instance types
    instance_of = []
    if 'P31' in claims:
      for p31 in claims['P31']:
        try:
          instance_of_qid = p31['mainsnak']['datavalue']['value']['id']
          if (os.path.exists(f"./cache/{instance_of_qid}")):
            file = open(f"./cache/{instance_of_qid}", "r")
            instance_dict = json.loads(file.read())
            file.close()
          else:
            instance_dict = get_entity_dict_from_api(instance_of_qid)
            file = open(f"./cache/{instance_of_qid}", "w")
            file.write(json.dumps(instance_dict))
            file.close()
          #instance_dict = get_entity_dict_from_api(instance_of_qid)
          instance_value = instance_dict['labels']['en']['value']
          instance_of.append(instance_value)
          del instance_dict
        except:
          None
    if len(instance_of) == 0:
      instance_of.append("unknown")

    # parse analytics
    pageview_count = 0
    registered_contributors_count = 0
    anonymous_contributors_count = 0
    num_wikipedia_lang_pages = 0
    description_300 = ""

    task1_urls = []
    task2_urls = []
    task3_urls = []
    task4_urls = []
    num_wikipedia_lang_pages = len(qwikidata_dict['sitelinks'].keys())
    for sitelink_key in qwikidata_dict['sitelinks'].keys():
      if not (sitelink_key == "enwiki"):
        continue
      sitelink_url = qwikidata_dict['sitelinks'][sitelink_key]['url']
      wiki_subdomain = urlparse(sitelink_url).hostname.split('.')[0]
      wiki_url_title = sitelink_url.split('/')[-1]

      url = f'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{wiki_subdomain}.wikipedia/all-access/user/{wiki_url_title}/daily/2022020100/2022022300'
      task1_urls.append(url)

      url = f'https://{wiki_subdomain}.wikipedia.org/w/api.php?action=query&prop=contributors&format=json&titles={wiki_url_title}'
      task2_urls.append(url)

      url = f'https://en.wikipedia.org/w/api.php?action=query&prop=extracts&explaintext&titles={wiki_url_title}&exchars=300&format=json'
      task3_urls.append(url)

      url = f'https://en.wikipedia.org/w/api.php?action=query&generator=images&titles={wiki_url_title}&prop=imageinfo&iiprop=url&format=json'
      task4_urls.append(url)

    task1_responses = asyncio.run(async_get_list_of_urls(task1_urls))
    task2_responses = asyncio.run(async_get_list_of_urls(task2_urls))
    task3_responses = asyncio.run(async_get_list_of_urls(task3_urls))
    task4_responses = asyncio.run(async_get_list_of_urls(task4_urls))
    for response in task1_responses:
      # count number of pageviews from 2022 February 1 to 2022 February 23 (by real human users)
      try:
        response_json = json.loads(response)
        for item in response_json['items']:
          pageview_count += item['views']
        del response_json
      except:
        None

    for response in task2_responses:
      # count number of contributors of this wiki page
      try:
        response_json = json.loads(response)
        try:
          registered_contributors_count += len(
            response_json['query']['pages'][list(response_json['query']['pages'].keys())[0]]['contributors'])
        except:
          None
        try:
          anonymous_contributors_count += \
            response_json['query']['pages'][list(response_json['query']['pages'].keys())[0]]['anoncontributors']
        except:
          None
        del response, response_json
      except:
        None

    for response in task3_responses:
      # retrieve description
      try:
        response_json = json.loads(response)
        try:
          description_300 = response_json['query']['pages'][list(response_json['query']['pages'].keys())[0]]['extract']
          description_300 = description_300.replace(',', '').replace(';', '').replace('\n', '')
        except:
          None
        del response, response_json
      except:
        None

    image_urls = []
    for response in task4_responses:
      # retrieve image urls
      try:
        response_json = json.loads(response)
        try:
          for img_entry in response_json['query']['pages']:
            title = response_json['query']['pages'][img_entry]['title']
            if not (title.endswith(".jpg") or title.endswith(".png")):
              continue
            image_url = response_json['query']['pages'][img_entry]['imageinfo'][0]['url']
            image_urls.append(image_url)
        except:
          None
        del response, response_json
      except:
        None
    del task1_responses, task2_responses, task3_responses, task4_responses

    # check if has image
    try:
      image = len(image_urls) > 0
      has_image = True
    except:
      has_image = False

    # load all images
    # if len(image_urls) > 3:
    #   image_urls = image_urls[0:3]
    image_responses = asyncio.run(async_get_list_of_urls(image_urls))
    image_filenames = []
    for i in range(0, len(image_urls)):
      if image_responses[i] == None:
        continue
      image_filename = os.path.basename(urlparse(image_urls[i]).path)
      if len(image_filename) > 30:
        image_filename = image_filename[len(image_filename) - 30 : len(image_filename)]
      image_filenames.append(image_filename)
      file = open(f"./images/{image_filename}", "wb")
      try:
        file.write(image_responses[i])
      except:
        None
      file.close()
    image_filenames_str = ','.join(image_filenames)

    line = f"{wikidata_value};{cityName};{lat};{lng};{is_shop};{is_tourism};{is_leisure};{is_other};{','.join(instance_of)};{pageview_count};{registered_contributors_count};{anonymous_contributors_count};{num_wikipedia_lang_pages};{has_image};{description_300};{image_filenames_str}"
    lines.append(line)
    del qwikidata_dict, image_responses
    # gc.collect()
    # print(line)

  lines = list(dict.fromkeys(lines))  # remove duplicates
  return lines

async def get_list_of_urls(url, session):
  try:
    url_md5 = hashlib.md5(url.encode()).hexdigest()
    if os.path.exists(f"./cache/{url_md5}"):
      file = open(f"./cache/{url_md5}", "rb")
      content_bytes = file.read()
      file.close()
      try:
        content_str = content_bytes.decode("utf-8")
        return content_str
      except Exception as e:
        return content_bytes
    else:
      async with session.get(url=url, headers={'User-agent': 'Mozilla/5.0'}) as response:
        resp = (await response.read())
      try:
        resp = resp.decode("utf-8")
        file = open(f"./cache/{url_md5}", "w") # will write string
      except Exception as e:
        file = open(f"./cache/{url_md5}", "wb") # will write bytes
      file.write(resp)
      file.close()
      return resp
  except Exception as e:
    if (url.endswith(".jpg") or url.endswith(".png")):
      print(e)
    None


async def async_get_list_of_urls(urls):
  async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(verify_ssl=False)) as session:
    ret = await asyncio.gather(*[get_list_of_urls(url, session) for url in urls])
  return ret



def get_request_cached(url):
  url_md5 = hashlib.md5(url.encode()).hexdigest()
  if(os.path.exists(f"./cache/{url_md5}")):
    file = open(f"./cache/{url_md5}", "r")
    content = file.read()
    file.close()
    return content, 200
  response = requests.get(url)
  if response.status_code == 200:
    file = open(f"./cache/{url_md5}", "w")
    file.write(response.text)
    file.close()
  return response.text, response.status_code