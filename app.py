from flask import Flask, render_template, request
from collections import OrderedDict
import requests
import json
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
    for game in games:
        payload = {'search': game}

        url = "https://api.rawg.io/api/games?key=c3f3802ae5a145e68a6bc2a687e8f25f"

        r = requests.get(url, headers=headers, params=payload)

        data = json.loads(r.text)
        for gamenm in data['results']:
            if game == gamenm['name']:
                # add logic here for if api cannot return any image
                # was thinking we use our logo (in github already)
                if gamenm['background_image'] == None:
                   dictgame[gamenm['name']] = 'ggslogo.png'
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

    r2 = requests.get(getid, headers=headers)

    data = json.loads(r2.text)

    app_ids = {}
    temp = data['applist']

    for id in temp['apps']:
        if id['name'] in games:
            app_ids[id['name']] = id['appid']

    sorted_ids = OrderedDict([(key, app_ids[key]) for key in games])
    id_to_link = []
    for id in sorted_ids.values():
        id_to_link.append(id)

    games_links = games
    # create https for game
    link_start = "https://store.steampowered.com/app/"
    for i in range(len(games)):
        games_links[i] = link_start + str(id_to_link[i]) + "/" + games_links[i].replace(" ", "_") + "/"
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


# default games
games_names, games_images = get_images(['RIFT', 'Rust', 'Quake Champions', 'Pummel Party', 'Fall Guys: Ultimate Knockout'])
games_links = get_links(['RIFT', 'Rust', 'Quake Champions', 'Pummel Party', 'Fall Guys: Ultimate Knockout'])

@app.route('/')
def homepage():
    return render_template("index.html", dgames=games_images, ngames=games_names, lgames=games_links)


@app.route('/recs', methods=['POST'])
def recs():
    username = request.form['username']
    g_names, g_images = get_user_recs('recommendations.csv', username)
    return render_template("index.html", dgames=g_images, ngames=g_names, lgames=games_links)


@app.route('/thanks')
def special_thanks():
    return render_template("special_thanks.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
