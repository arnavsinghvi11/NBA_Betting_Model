import date
import helper_functions
import re
import pandas as pd
from unidecode import unidecode

class BoxScore:
    date = date.Date()
    def extract_stat(self, tbl_input, stat):
        #extract numerical value for given statistic
        return tbl_input.split(stat)[1].split('</td><td')[0]
    
    def extract_box_score_stats(self, tbls_input, advanced_condition, team, opponent, homecourt_advantage, stats_list):
        #extract statistics from box score of given matchup
        data = []

        #iterate through list of players with minutes played
        for i in tbls_input.split('href="/players/')[1:]:
            i = i.split('scope="row"')[0]

            #extract statistics from specified box score table type  
            if advanced_condition:
                condition_check = 'Team' not in i and len(i.split('.html">')) > 1
            else:
                condition_check = 'Team' not in i
            if condition_check:
                name = i.split('.html">')[1].split('</a>')[0]
                if team:
                    data_point = [name, team, opponent, homecourt_advantage]
                else:
                    data_point = [name]
                if all([stat.split('">')[0] in i for stat in stats_list]):
                    for stat in stats_list:
                        if 'bpm' in stat:
                            data_point.append(i.split('bpm')[1].split('">')[1].split('</td></tr')[0])
                        else:
                            data_point.append(self.extract_stat(i, stat))
                    data.append(data_point)
        return data

    def extract_box_score(self, box_score_type, soup, team_tbl, team, opponent, homecourt_advantage):
        #determine between basic and advanced box score table to extract statistics from 
        if 'Basic' in box_score_type:
            tbls = []
            for k in team_tbl.split(box_score_type):
                tbls.append(k.split('section_wrapper')[0])
            tbls = [i for i in tbls if "csk" in i]
            
            #basic box score table statistics
            stats_list = ['mp">', 'pts">','ast">', 'trb">','stl">','blk">','fg3">']
            return self.extract_box_score_stats(tbls[0], False, team, opponent, homecourt_advantage, stats_list)
        else:
            data = []

            #advanced box score table statistics
            stats_list = ['ts_pct">','efg_pct">','fg3a_per_fga_pct">','orb_pct">','drb_pct">','trb_pct">','ast_pct">','stl_pct">','blk_pct">','usg_pct">','off_rtg">','def_rtg">','bpm">']
            for i in str(soup).split(box_score_type)[1:]:
                data.append(self.extract_box_score_stats(i, True, None, None, None, stats_list))
            return data 

    def homecourt_advantage(self, team, homecourt):
        #determine homecourt advantage of game for each player
        if team == homecourt:
            return 1
        return 0

    def full_box_scores(self, month, day):
        #extract yesterday's box score statistics
        total_site_data = helper_functions.site_scrape('https://www.basketball-reference.com/boxscores/' + '?month=' + month + '&day=' + day + '&year=2023')
        
        #iterate through matchups list to add each game's box score link 
        links = []
        for i in str(total_site_data).split('<p class="links"><a href')[1:]:
            links.append('https://www.basketball-reference.com' + i.split('Box')[0].split('"')[1])
        data, advanced_data = [], []

        #iterate through each game's box score and add basic and advanced statistics for players
        for link in links:
            site_data = helper_functions.site_scrape(link)
            team1 = str(site_data).split('Basic and Advanced Stats Table</caption>')[0].split('sortable stats_table')[1].split('caption>')[1].strip()
            team2 = str(site_data).split('Basic and Advanced Stats Table</caption>')[1].split('Basic and Advanced Stats')[-1].split('caption>')[1].strip()
            homecourt = re.findall("<strong>[A-z0-9\s]+at[A-z0-9\s]+Box Score,", str(site_data))[0].split(' at ')[1].split(' Box')[0]
            if not data:
                data = self.extract_box_score('Basic Box Score Stats', site_data, str(site_data).split('Basic and Advanced Stats Table</caption>')[1:][0], team1, team2, self.homecourt_advantage(team1, homecourt))
            else:
                data.extend(self.extract_box_score('Basic Box Score Stats', site_data, str(site_data).split('Basic and Advanced Stats Table</caption>')[1:][0], team1, team2, self.homecourt_advantage(team1, homecourt)))
            data.extend(self.extract_box_score('Basic Box Score Stats', site_data, str(site_data).split('Basic and Advanced Stats Table</caption>')[1:][1], team2, team1, self.homecourt_advantage(team2, homecourt)))
            if not advanced_data:
                advanced_data = self.extract_box_score('>Advanced Box Score Stats<', site_data, None, None, None, None)[0]
            else:
                advanced_data.extend(self.extract_box_score('>Advanced Box Score Stats<', site_data, None, None, None, None)[0])
            advanced_data.extend(self.extract_box_score('>Advanced Box Score Stats<', site_data, None, None, None, None)[1])
        statistics = pd.DataFrame (data, columns = ['name', 'team', 'opp', 'hmcrt_adv', 'mp', 'pts', 'ast', 'trb', 'stl', 'blk', 'fg3',])
        advanced_statistics = pd.DataFrame (advanced_data, columns = ['name', 'ts_pct', 'efg_pct', 'fg3a_per_fga_pct', 'orb_pct', 'drb_pct', 'trb_pct', 'ast_pct', 'stl_pct', 'blk_pct', 'usg_pct', 'off_rtg', 'def_rtg', 'bpm'])
        box_score = pd.merge(statistics, advanced_statistics, left_on = 'name', right_on = 'name')
        box_score['mp'] = box_score['mp'].str.replace(':', '.')
        return box_score

    def update_all_box_score_results(self,  date, month, day):
        #returns cumulative box score statistics for players
        all_box_score_results = pd.read_csv("/work/All_Box_Score_Results.csv")
        box_score = pd.read_csv("/work/NBA-Bets-Box-Score-Results/" + "NBA-Bets-Box-Score-Results-" + month + "-" + day +".csv")
        box_score['date'] = month + "-" + day
        box_score['name'] = box_score['name'].apply(unidecode)
        box_score['name'] = box_score['name'].apply(helper_functions.abbrv)
        box_score['date'] = box_score['date'].apply(date.date_converter)
        all_box_score_results = all_box_score_results.append(box_score)
        all_box_score_results.to_csv("/work/All_Box_Score_Results.csv")
        
        #formatting for names and dates for predictions/evaluations 
        all_box_score_results['date'] = all_box_score_results['date'].astype('int')
        all_box_score_results = all_box_score_results.sort_values(by=['date'], ascending = False)
        all_box_score_results.set_index(['name','team']).index.factorize()[0]+1
        all_box_score_results = all_box_score_results.drop_duplicates(['date','name','team', 'opp', 'hmcrt_adv', 'pts', 'ast', 'trb'], keep='last')
        return all_box_score_results