from urllib.error import URLError
from urllib.request import urlopen
import os
import requests


def load_data_factory(url, dest, human_name, read_func):
    def load_data():
        if not os.path.exists(dest):
            download_resource(url, dest, human_name)
        data = read_func(dest)
        return data

    return load_data


import wget


def download_resource(url, path, name):
    attempts = 0
    success = False
    print("Downloading resource {}, this can take a moment...".format(name))
    while attempts < 3:
        try:
            r = requests.get(url)
            with open(path, "wb") as f:
                f.write(r.content)
            # response = urlopen(url, timeout=5)
            # content = response.read()
            # f = open(path, "wb")
            # f.write(content)
            # f.close()
            # wget.download(url,path)
            success = True
            break
        except URLError as e:
            attempts += 1
            print(type(e))
    if not success:
        print(
            "Failed to download {}. \n".format(name)
            + "Try later to download it manually from {}".format(url)
            + "\n and put it in ssm/resources"
        )
