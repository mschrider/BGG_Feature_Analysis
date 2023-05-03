import pandas as pd
import math
import urllib3
import numpy as np
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
import time
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import shap
import re

pd.set_option('display.max_columns', None)

def get_top_boardgame_ids(num_top: int = 2000, page_start: str = '1'):
    # Scrape Information from Board Game Geek (BGG)
    # Set Options
    options = webdriver.ChromeOptions()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--incognito')

    # Initiate the driver (chromedriver needed if not downloaded)
    driver = webdriver.Chrome(r"C:\Program Files (x86)\Chrome Driver\chromedriver", chrome_options=options)
    driver.get("https://boardgamegeek.com/browse/boardgame/page/%s" % page_start)

    WebDriverWait(driver, 5).until(
        lambda s: s.find_element(By.CLASS_NAME, 'legacy').is_displayed()
    )
    # logging in by hand to avoid the 2k cap on pulls, tried getting selinium to do it but wanted to move on
    time.sleep(15)

    # TODO: (Future Work)Get selenium to input credentials to avoide the 2k cap on pulls
    # wait = WebDriverWait(driver, 30)
    # wait.until(EC.element_to_be_clickable((By.CLASS_NAME, 'btn btn-sm'))).click()
    # wait.until(EC.element_to_be_clickable((By.ID, 'inputUsername'))).send_keys("mschrider")
    # wait.until(EC.element_to_be_clickable((By.ID, 'inputPassword'))).send_keys("pcXhbtGqaSBGpH&Qwe73")
    # wait.until(EC.element_to_be_clickable((By.LINK_TEXT, 'btn btn-primary'))).click()

    # Get Game ID to feed into BGG XML API
    top_games = []
    i = 1
    while i < math.ceil(num_top / 100) + 1:
        try:
            WebDriverWait(driver, 3).until(
                lambda s: s.find_element(By.ID, 'collection')
            )
        except TimeoutException:
            break

        soup = bs(driver.page_source, 'lxml')

        top_games.append(soup)

        if num_top > 100:
            load_more = driver.find_element(By.LINK_TEXT, '%i' % (i + 1))
            if load_more:
                driver.execute_script("arguments[0].click();", load_more)
        i += 1

    # Pull all the game IDs from top games into a list
    top_games_dict = {'id': [], 'rank': [], 'name': []}
    for g in top_games:
        all_game_info = g.find_all("tr", id="row_")
        for game in all_game_info:
            game_id_link = game.find("a", class_="primary")['href']
            game_id = game_id_link.split('/')[2]
            game_rank = game.find("td", class_="collection_rank").get_text(strip=True)
            game_name = game.find("a", class_="primary").get_text()
            top_games_dict['id'].append(game_id)
            top_games_dict['rank'].append(game_rank)
            top_games_dict['name'].append(game_name)

    top_games_df = pd.DataFrame(top_games_dict)
    top_games_df.set_index('id', inplace=True)
    top_games_df.to_csv(r'Data/game_ids/game_ids_%s.csv' % datetime.now().strftime('%Y-%m-%d_%H%M%S'))

    return top_games_df


def get_boardgame_info(game_ids: list):
    game_stats = []
    # BGG XML API will only return 100 games at a time
    chunked_ids = [game_ids[i:i + 100] for i in range(0, len(game_ids), 100)]
    i = 0
    current_time = datetime.now().strftime('%Y-%m-%d_%H%M%S')
    # Loop through entire list of game IDs 100 at a time to gather game data
    for ids in chunked_ids:
        cs_game_ids = ','.join([str(x) for x in ids])
        url = 'http://www.boardgamegeek.com/xmlapi/boardgame/%s?stats=1' % cs_game_ids
        http = urllib3.PoolManager()
        page = http.request('GET', url)

        soup = bs(page.data, 'lxml')
        game_stats.append(soup)

        # write to file to avoid having to pull every time
        with open('Data/game_info_html/game_info_%s.html' % current_time,
                  'a',
                  encoding='utf-8') as f:
            f.write(str(soup))
        i += 1
        print(f"{i} of {len(chunked_ids)} Chunks Complete")

    return game_stats


def parse_game_info(game_info):
    # TODO: (Future work)check if expansion
    # TODO: (Future work)check number of expansions
    # TODO: (Future work)check number of versions
    # TODO: (Future work)Anything to do with family?
    game_info_dict = {}
    # break into the chunks of boardgames
    if type(game_info) == str:
        game_info = bs(game_info, "xml")
    for s in game_info:
        all_game_info = s.find_all('boardgame')
        # for each game in the boardgame chunk, retrieve the specified data and add to dictionary
        for game in all_game_info:
            id = game['objectid']

            if game.find('name', primary="true"):
                title = game.find('name', primary="true").get_text(strip=True)
            else:
                continue

            if game.yearpublished:
                year_published = game.yearpublished.string
            else:
                year_published = None

            if game.minplayers:
                min_players = game.minplayers.string
            else:
                min_players = None

            if game.maxplayers:
                max_players = game.maxplayers.string
            else:
                max_players = None

            if game.playingtime:
                play_time = game.playingtime.string
            else:
                play_time = None

            if game.minplaytime:
                pt_min = game.minplaytime.string
            else:
                pt_min = None

            if game.maxplaytime:
                pt_max = game.maxplaytime.string
            else:
                pt_max = None

            if game.age:
                age = game.age.string
            else:
                age = None

            categories = []
            for category in game.find_all('boardgamecategory'):
                categories.append(category.string)

            mechanics = []
            for mechanic in game.find_all('boardgamemechanic'):
                mechanics.append(mechanic.string)

            if game.boardgamesubdomain:
                subdomain = game.boardgamesubdomain.string
            else:
                subdomain = None

            num_honors = len(game.find_all('boardgamehonor'))
            if game.bayesaverage:
                rating = game.bayesaverage.string
            else:
                rating = None

            if game.averageweight:
                complexity = game.averageweight.string
            else:
                complexity = None
            game_info_dict[id] = {'title': title, 'rating': rating, 'year_published': year_published,
                                  'min_players': min_players,
                                  'max_players': max_players, 'play_time': play_time, 'pt_min': pt_min,
                                  'pt_max': pt_max, 'age': age, 'complexity': complexity, 'categories': categories,
                                  'mechanics': mechanics,
                                  'subdomain': subdomain, 'num_honors': num_honors}
    # transform the filled dictionary into a pandas dataframe
    game_info_df = pd.DataFrame.from_dict(game_info_dict, orient='index')
    # save the dataframe as a CSV to reference
    game_info_df.to_csv(r'Data/game_info_dfs/game_info_df_%s.csv' % datetime.now().strftime('%Y-%m-%d_%H%M%S'))

    return game_info_df


# Create wrapper function to gather the data, either from scraping/API or from files
def gather_data(scrape_ids=False, scrape_info=False, num_games=25000):
    if scrape_ids:
        game_ids = get_top_boardgame_ids(num_games)  # only ~25k games have ratings, so good limiter
    else:
        game_ids = pd.read_csv(r'Data/game_ids/game_ids_2023-04-23_155856.csv', index_col='id')  # csv of top 25000 game ids

    if scrape_info:
        game_info = get_boardgame_info(game_ids.index.to_list())
    else:
        with open(r'Data/game_info_html/game_info_2023-04-23_170007.html', 'r', encoding='utf-8') as f:  # html of top games
            game_info = f.read()

    parsed_game_data = parse_game_info(game_info)

    parsed_game_data = game_ids[['rank']].join(parsed_game_data, how='right')

    return parsed_game_data


def preprocess_game_info(game_info_df: pd.DataFrame):
    # high level clean up
    pp_df = game_info_df.reset_index(drop=True).copy()
    # dropping rank, num_honors as they are closer to targets and decided to use rating as the only target
    pp_df.drop(columns=['rank', 'num_honors'], inplace=True)
    # cleaning up datatypes where needed
    pp_df = pp_df.astype({'rating': 'float',
                                      'year_published': 'int',
                                      'min_players': 'int',
                                      'max_players': 'int',
                                      'play_time': 'int',
                                      'pt_min': 'int',
                                      'pt_max': 'int',
                                      'age': 'int',
                                      'complexity': 'float'})
    pp_df['categories'] = pp_df['categories'].apply(lambda x: ['category: ' + c for c in x])
    pp_df['mechanics'] = pp_df['mechanics'].apply(lambda x: ['mechanic: ' + c for c in x])

    # one-hot encode categorical columns
    # decided to treat players, play time, and age as values instead of categories

    # set up the multilabel binarizers for multi-entry columns (categories, mechanics)
    mlb_cat = MultiLabelBinarizer(sparse_output=True)
    mlb_mech = MultiLabelBinarizer(sparse_output=True)
    # execute one-hot encoding against the multi-entry categorical columns
    pp_df = pp_df.join(
        pd.DataFrame.sparse.from_spmatrix(mlb_cat.fit_transform(pp_df.pop('categories')), index=pp_df.index,
                                          columns=mlb_cat.classes_))

    # combine categories that make up less than x% of the data into "category: other"
    x = 0.05  # minimum percentage of total desired for categorical column
    min_num = x * len(pp_df.index)
    cat_to_other = pp_df[mlb_cat.classes_].sum() <= min_num  # identify columns with too few entries
    # create "category: other", fill in if any of the identified categories are present
    pp_df['category: Other'] = pp_df[cat_to_other.index[cat_to_other]].agg('max', axis=1)
    pp_df.drop(columns=cat_to_other.index[cat_to_other], inplace=True)

    pp_df = pp_df.join(
        pd.DataFrame.sparse.from_spmatrix(mlb_mech.fit_transform(pp_df.pop('mechanics')), index=pp_df.index,
                                          columns=mlb_mech.classes_))
    # combine mechanics that make up less than y% of the data into "mechanics: other"
    y = 0.05  # minimum percentage of total desired for categorical column
    min_num = x * len(pp_df.index)
    mech_to_other = pp_df[mlb_mech.classes_].sum() <= min_num # identify columns with too few entries
    # create "category: other", fill in if any of the identified categories are present
    pp_df['mechanic: Other'] = pp_df[mech_to_other.index[mech_to_other]].agg('max', axis=1)
    pp_df.drop(columns=mech_to_other.index[mech_to_other], inplace=True)

    # prep year for one-hot encoding
    decade_bins = [0, 1980, 1990, 2000, 2010, 2020, 10000]
    decade_labels = ['Before 1980', '1980s', '1990s', '2000s', '2010s', 'After 2020']
    pp_df['decade_published'] = pd.cut(x=pp_df['year_published'], bins=decade_bins, labels=decade_labels)
    pp_df.drop(columns=['year_published'], inplace=True)
    # execute one-hot encoding on decades and subdomain
    ohe = OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=y, sparse_output=False)
    columns_to_ohe = ['subdomain', 'decade_published']
    one_hot_array = ohe.fit_transform(pp_df.loc[:, columns_to_ohe])
    one_hot_df = pd.DataFrame(one_hot_array, columns=ohe.get_feature_names_out())
    pp_df = pd.concat([pp_df, one_hot_df], axis=1)
    pp_df.drop(columns=columns_to_ohe, inplace=True)

    return pp_df

def model_linear(preprocessed_df: pd.DataFrame):
    # bring in the data and set up train/test split
    lr_df = preprocessed_df.copy()
    lr_df['rating'] = round(lr_df['rating'], 2)
    X = lr_df.drop(columns=['title', 'rating'])
    y = lr_df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Initialize and fit the model
    LR = LinearRegression()
    LR.fit(X_train, y_train)

    return LR.score(X_test, y_test)


def model_tree(preprocessed_df: pd.DataFrame):
    # bring in the data and set up train/test split
    dt_df = preprocessed_df.copy()
    dt_df['rating'] = round(dt_df['rating']*2)/2
    dt_df['rating'] = dt_df['rating'].astype('str')
    X = dt_df.drop(columns=['title', 'rating'])
    y = dt_df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Initialize and fit the model
    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)

    # generate feature importance -- as this model proved insufficient, did not implement SHAP here
    importance = clf.feature_importances_

    for i, v in enumerate(importance):
        print(f'Feature: {i}, Score: {v}')

    return clf.score(X_test, y_test), importance


def model_forest(preprocessed_df: pd.DataFrame):
    # bring in the data and set up train/test split
    dt_df = preprocessed_df.copy()
    dt_df['rating'] = round(dt_df['rating']*2)/2  # rounding to .5
    dt_df['rating'] = dt_df['rating'].astype('str')
    X = dt_df.drop(columns=['title', 'rating'])
    y = dt_df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

    # Initialize and fit the model
    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)

    # get accurace and importances
    score = rfc.score(X_test, y_test)
    importances = pd.Series(rfc.feature_importances_, index=X.columns)

    # use SHAP to visualize importance of features through .TreeExplainer
    rf_explainer = shap.TreeExplainer(rfc)

    rf_shap_values = np.array(rf_explainer.shap_values(X_test))  # kwarg approximate=True for testing, much faster
    shap.summary_plot(rf_shap_values[11], features=X_test, feature_names=X.columns, show=False)
    plt.title("8.5 Class SHAP Summary")
    plt.savefig("shap_summary_8_5_class.svg", dpi=700)
    plt.close()

    index_to_highlight = y_test[y_test == '8.5'].index[0]
    name_highlighted = re.sub('[^0-9a-zA-Z]+', '', dt_df['title'].loc[index_to_highlight])
    shap.force_plot(rf_explainer.expected_value[1],
                    rf_explainer.shap_values(X_test.loc[[0]])[1],
                    X_test.loc[[0]],
                    matplotlib=True,
                    show=False)
    plt.title(f"{name_highlighted} Force Plot")
    plt.savefig(f"{name_highlighted}_force_plot.svg", dpi=700)
    plt.close()

    return score, importances, rf_shap_values


if __name__ == '__main__':

    pre_p_df = preprocess_game_info(gather_data())
    print(pre_p_df.size)

    rf_score, rf_importances, rf_shapvalues = model_forest(pre_p_df)
    dt_score, dt_importances = model_tree(pre_p_df)
    lr_score = model_linear(pre_p_df)
