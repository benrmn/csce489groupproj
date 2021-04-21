from flask import Flask, render_template, request
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
                if gamenm['background_image'] == '404':
                   dictgame[gamenm['name']] = 'ggslogo.png'
                else:
                   dictgame[gamenm['name']] = gamenm['background_image']

                # dictgame[gamenm['name']] = gamenm['background_image']

    games_names = list(dictgame)
    vals = dictgame.values()
    games_images = list(vals)

    return games_names, games_images


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
    return render_template("index.html", dgames=games_images, ngames=games_names)


@app.route('/recs', methods=['POST'])
def recs():
    username = request.form['username']
    g_names, g_images = get_user_recs('recommendations.csv', username)
    return render_template("index.html", dgames=g_images, ngames=g_names)


@app.route('/thanks')
def special_thanks():
    return render_template("special_thanks.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
