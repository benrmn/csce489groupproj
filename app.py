from flask import Flask, render_template, request
from collections import OrderedDict
import numpy as np
import requests
import json
import copy
import csv
import urllib3

app = Flask(__name__)


def get_images(games):
    headers = {
        'User-Agent': 'csce 489 ggs project',
        'From': 'bramon24@tamu.edu',
    }

    # games = ['RIFT', 'Rust', 'Quake Champions', 'Pummel Party', 'Fall Guys: Ultimate Knockout']

    dictgame = {}
    temp_name = ""
    temp_games = games
    for game in temp_games:
        if game == 'Counter-Strike Global Offensive':
            temp_name = 'Counter-Strike: Global Offensive'
        else:
            temp_name = game

        payload = {'search': temp_name}

        url = "https://api.rawg.io/api/games?key=c3f3802ae5a145e68a6bc2a687e8f25f"

        r = requests.get(url, headers=headers, params=payload)

        data = json.loads(r.text)

        temp = []
        for gamenm in data['results']:
            temp.append(gamenm['name'])

        if temp_name not in temp:
            dictgame[temp_name] = 'https://i.ibb.co/3fxbY3D/ggslogo.jpg'
        else:
            for gamenm in data['results']:
                if temp_name == gamenm['name']:
                    # add logic here for if api cannot return any image
                    # was thinking we use our logo (in github already)
                    if not str(gamenm['background_image']):
                        dictgame[gamenm['name']] = 'check'
                    else:
                        dictgame[gamenm['name']] = gamenm['background_image']
                    # dictgame[gamenm['name']] = gamenm['background_image']

    games_names = list(dictgame)
    vals = dictgame.values()
    games_images = list(vals)

    return games_names, games_images


def get_links(games):

    headers = {
        'User-Agent': 'csce 489 ggs project',
        'From': 'bramon24@tamu.edu',
    }
    getid = "https://api.steampowered.com/ISteamApps/GetAppList/v2/"
    link_start = "https://store.steampowered.com/app/"

    r2 = requests.get(getid, headers=headers)

    data = json.loads(r2.text)

    app_ids = {}
    temp = data['applist']
    games_links = np.empty(len(games), dtype=object)
    idx = 0
    for id in temp['apps']:
        if id['name'] in games or id['name'].replace(':', '') in games:
            for i in range(len(games)):
                if games[i] == id['name'] or games[i] == id['name'].replace(':', ''):
                    idx = i
            games_links[idx] = link_start + str(id['appid']) + "/"

    for i in range(len(games_links)):
        if type(games_links[i]) == float:
            games_links[i] = link_start

    print(games_links)
    return games_links


def get_user_recs(filename, username):
    top_5_recs = []
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            if row[0] == username:
                for i in range(1, 6):
                    print(row[i])
                    top_5_recs.append(row[i])
                break
        g_name, g_image = get_images(top_5_recs)
        return g_name, g_image


def get_most_popular():
    with open('recommendations.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        top_popular = []
        for row in csv_reader:
            for i in range(1, 6):
                top_popular.append(row[i])
            return get_images(top_popular)


@app.route('/')
def homepage():
    games_names, games_images = get_most_popular()
    temp_names = games_names
    games_links = get_links(games_names)
    return render_template("index.html", dgames=games_images, ngames=temp_names, lgames=games_links)


@app.route('/recs', methods=['POST'])
def recs():
    username = request.form['username']
    g_names, g_images = get_user_recs('recommendations.csv', username)
    temp_names = g_names
    games_links = get_links(g_names)
    return render_template("index.html", dgames=g_images, ngames=temp_names, lgames=games_links)


@app.route('/thanks')
def special_thanks():
    return render_template("special_thanks.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
