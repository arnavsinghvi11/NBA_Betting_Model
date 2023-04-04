from bs4 import BeautifulSoup
import chromedriver_autoinstaller
import date
import datetime
from datetime import datetime
from datetime import timedelta
import pytz
from pytz import timezone
import os
import numpy as np
import pandas as pd
import regex as re
import requests
import selenium
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
import time


class Bets:
    date = date.Date()

    def site_scrape_chrome(self, url):
        # initiating webdriver settings for Google Chrome
        chromedriver_autoinstaller.install()
        os.environ["LANG"] = "en_US.UTF-8"
        options = Options()
        options.headless = True
        options.add_argument("disable-infobars")
        options.add_argument("--disable-extensions")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--no-sandbox")
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        driver = Chrome(options=options)
        tz_params = {'timezoneId': 'America/Los_Angeles'}
        driver.execute_cdp_cmd('Emulation.setTimezoneOverride', tz_params)
        driver.get(url)
        time.sleep(5)
        html = driver.page_source

        # apply BeautifulSoup to scrape web page contents
        soup = BeautifulSoup(html, "html.parser")
        driver.close()  # closing the webdriver
        return soup

    def schedule(self, date):
        #returning time 10 mins before earliest scheduled NBA game for today
        today_date = date.date_formatting(0)
        today_date = today_date.split(' ')[0].split('/')[2] + today_date.split(
            ' ')[0].split('/')[0] + today_date.split(' ')[0].split('/')[1]
        tmrw_date = date.date_formatting(1)
        tmrw_date = tmrw_date.split(' ')[0].split('/')[2] + tmrw_date.split(
            ' ')[0].split('/')[0] + tmrw_date.split(' ')[0].split('/')[1]
        soup = self.site_scrape_chrome("https://www.espn.com/nba/schedule")
        
        #scrape game times before tomorrow's game and return in Pacific Standard Time
        tdaycontent = ''
        for x in str(soup).split(str(today_date))[1:]:
            tdaycontent = tdaycontent + str(x)
        for i in tdaycontent.split(tmrw_date)[1].split('PM EST'):
            earliest = i.split(' at ')[1]
            break
        pacific_time = datetime.strptime(
            (str(int(earliest.split(':')[0]) + 12) + ':' +
             earliest.split(':')[1]).strip().split(' ')[0], "%H:%M") - timedelta(hours=3,
                                                                   minutes=10)
        running_time = str(pacific_time.hour) + ':' + \
            str(pacific_time.minute) + ':00'
        print(running_time)
        return running_time

    def preprocessing(self, entry, name, team, net_unit):
        #data preprocessing of predictions
        prediction = False
        if len(entry.split('"play":')) > 1:
            prediction = entry.split('"play":')[1].split(',')[0].replace(
                '"', '')
        
        #extract expert's username
        if name:
            expert = name
        else:
            expert = entry.split(',"picture_url":')[0].replace('"', '')
        
        #extract current bet's live odds (measure of how much you can win from bet)
        odds = entry.split('"odds":')[1].split(',')[0]

        #extract current bet's recommended units (measure of how much of standard entry to place on current bet)
        units = entry.split('"units":')[1].split(',')[0]
        
        #extract current expert's cumulative betting wins and losses
        if net_unit:
            units_net_record = net_unit
        else:
            units_net_record = 0

        #extract multiplier for current payout (formulas derived from US betting odds)
        if int(odds) > 0:
            multiplier = (int(odds) / 100) + 1
        else:
            multiplier = (-100 / int(odds)) + 1
        if prediction:
            if re.match(r"[A-Z]\.", prediction):
                bet_entry = []
                bet_entry.append(prediction)
                bet_entry.append(team)
                if '}]' in expert:
                    expert = expert.split('}]')[0]
                bet_entry.append(expert)
                bet_entry.append(odds)
                bet_entry.append(float(units))
                bet_entry.append(float(multiplier * float(units)))
                bet_entry.append(units_net_record)
                return bet_entry

    def output(self, all_box_score_results):
        #extract bets predictions for today's games
        soup = self.site_scrape_chrome(
            "https://www.actionnetwork.com/nba/picks")
        live_bets = str(soup).split('"status":"complete"')[0]
        
        #extract home and away teams for each matchup
        games = []
        for i in live_bets.split('"game-picks-header__teams">')[1:]:
            two_teams = i.split('Picks</div>')[0]
            away_team = two_teams.split('<!-- --> ')[0]
            home_team = two_teams.split('@ <!-- -->')[1].split('<!-- -->')[0]
            games.append([away_team, home_team])
        bet_entries = []
        #extract current bets for games to be played 
        for x in live_bets.split('"real_status":"scheduled","status_display":null,"start_time"')[1:]:
            team = []
            teams = x.split('"full_name":"')
            for t in teams[1:3]:
                team.append(t.split('","display_name')[0])
            x = re.sub('"user_mentions":.*?]', '', x)
            for i in x.split('"username":')[1:]:
                if '},{' in i and 'play' in i:
                    for j in i.split('},{'):
                        if 'units_net' in j:
                            if 'First' not in j:
                                if "odds:" in j:
                                    if len(j.split('"record":')) > 1:
                                        bet_entries.append(
                                            self.preprocessing(
                                                j,
                                                i.split(',"picture_url":')
                                                [0].replace('"', ''), team,
                                                i.split('"record":')[1].split(
                                                    '"units_net":')[1].split(
                                                        ',')[0]))
                                    else:
                                        bet_entries.append(
                                            self.preprocessing(
                                                j,
                                                i.split(',"picture_url":')
                                                [0].replace('"', ''), team, None))
                        else:
                            if 'units_net' in i and 'play' in i and 'settled_at":null' in i and 'First' not in i:
                                bet_entries.append(
                                    self.preprocessing(i, None, team, None))
                else:
                    if 'play' in i:
                        if 'units_net' in i:
                            if 'First' not in i:
                                if "odds:" in j:
                                    if len(i.split('"record":')) > 1:
                                        bet_entries.append(
                                            self.preprocessing(
                                                i,
                                                i.split(',"picture_url":')
                                                [0].replace('"', ''), team,
                                                i.split('"record":')[1].split(
                                                    '"units_net":')[1].split(
                                                        ',')[0]))
                                    else:
                                        bet_entries.append(
                                            self.preprocessing(
                                                i,
                                                i.split(',"picture_url":')
                                                [0].replace('"', ''), team, None))
                        else:
                            if 'units_net' in i and 'play' in i and 'settled_at":null' in i and 'First' not in i:
                                bet_entries.append(
                                    self.preprocessing(i, None, team, None))

        
        #build dataset for today's predictions and remove any duplicate bet entries
        bets = pd.DataFrame(
            [i for i in bet_entries if i is not None],
            columns=[
                'Play', 'Teams', 'Expert', 'Odds', 'Units', 'Payout',
                'Net Units Record'
            ]).drop_duplicates(subset=['Play', 'Expert'],
                               keep='first').reset_index(drop=True)
        
        #determine variables for each bet - player's current team, matchup oppponent, matchup's homecourt advantage
        names, set_teams, opponents, hmcrt_advantages = [], [], [], []
        for i in range(len(bets)):
            bet = bets.loc[i]
            name = bet['Play'].split(' ')[0]
            print(name)
            print(bet['Teams'])
            if 'LA' in bet['Teams'][0]:
                bet['Teams'][0] =  bet['Teams'][0].replace('LA', 'Los Angeles')
            matching_name = all_box_score_results[all_box_score_results['name']
                                                  == name]
            all_games = matching_name[matching_name['team'].isin(bet['Teams'])]
            found_team = False
            if len(set(all_games['name'].values)) > 1:
                set_teams.append('')
                names.append(np.NAN)
                opponents.append(np.NAN)
                hmcrt_advantages.append(np.NAN)
            else:
                if name == 'R.Westbrook':
                    set_teams.append('')
                    names.append(np.NAN)
                    opponents.append(np.NAN)
                    hmcrt_advantages.append(np.NAN) 
                    print('skipped')
                    continue
                if len(set(all_games['name'].values)) == 0:
                    if len(set(matching_name['name'].values)) != 1:
                        set_teams.append('')
                        names.append(np.NAN)
                        opponents.append(np.NAN)
                        hmcrt_advantages.append(np.NAN)
                        continue
                    else:
                        found_team = True
                found_team = True
                if found_team:
                    set(all_games['team'].values)
                    plyr_team = list(set(all_games['team'].values))[0]
                    print(plyr_team)
                    for g in games:
                        if ((len(plyr_team.split(' '))) > 2 and
                            (plyr_team.split(' ')[1] + ' ' +
                             plyr_team.split(' ')[2]
                             == g[0])) or plyr_team.split(' ')[-1] == g[0]:
                            hmcrt_advantages.append(0)
                            for t in all_box_score_results['team'].values:
                                if g[1] in t:
                                    opponents.append(t)
                                    names.append(name)
                                    break
                            break
                        if ((len(plyr_team.split(' '))) > 2 and
                            (plyr_team.split(' ')[1] + ' ' +
                             plyr_team.split(' ')[2]
                             == g[1])) or plyr_team.split(' ')[-1] == g[1]:
                            hmcrt_advantages.append(1)
                            for t in all_box_score_results['team'].values:
                                if g[0] in t:
                                    opponents.append(t)
                                    names.append(name)
                                    break
                            break
                    set_teams.append(plyr_team)
        bets['name'] = names
        bets['Teams'] = set_teams
        bets['opponent'] = opponents
        bets['hmcrt_adv'] = hmcrt_advantages

        #determine profit based on betting payouts minus inputted bet entry unit
        bets['Profit'] = bets['Payout'] - bets['Units']
        bets = bets.dropna()
        return bets
