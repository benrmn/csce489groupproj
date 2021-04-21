from flask import Flask, render_template, request
import requests
import json

app = Flask(__name__)

headers = {
    'User-Agent': 'csce 489 ggs project',
    'From': 'bramon24@tamu.edu',
}

games = ['RIFT', 'Rust', 'Quake Champions', 'Pummel Party', 'Fall Guys: Ultimate Knockout']

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
            #if gamenm['background_image'] == None:
            #    dictgame[gamenm['name']] = 'ggslogo.png'
            #else:
            #    dictgame[gamenm['name']] = gamenm['background_image']
                
            dictgame[gamenm['name']] = gamenm['background_image']

games_names = list(dictgame)
vals = dictgame.values()
games_images = list(vals)


@app.route('/')
def homepage():
    return render_template("index.html", dgames=games_images, ngames=games_names)


@app.route('/recs', methods=['POST'])
def recs():
    username = request.form['username']
    return render_template("index.html", dgames=games_images, ngames=games_names)


@app.route('/thanks')
def special_thanks():
    return render_template("special_thanks.html")


@app.route('/about')
def about():
    return render_template("about.html")


if __name__ == '__main__':
    app.run()
