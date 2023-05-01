import datetime
from datetime import datetime
from datetime import timedelta
import pytz
from pytz import timezone

class Date:
    def date_formatting(self, days):
        #returns formatted version of date inputted to include month, date, year, current time
        date_format='%m/%d/%Y %H:%M:%S %Z'
        date = datetime.now(tz=pytz.utc)
        if days > 0: 
            correct_date = date.astimezone(timezone('US/Pacific')) + timedelta(days = 1)
        elif days < 0:
            correct_date = date.astimezone(timezone('US/Pacific')) - timedelta(days = 1)
        else:
            correct_date = date.astimezone(timezone('US/Pacific'))
        return correct_date.strftime(date_format)

    def date_month_day(self, days):
        #returns inputted date's month and day
        date_formatted = self.date_formatting(days)
        date_formatted = date_formatted.split('/')[0] + '/' + date_formatted.split('/')[1]
        return date_formatted.split('/')[0].lstrip('0'), date_formatted.split('/')[1].lstrip('0')

    def date_converter(self, col):
        #returns formatted version of date to maintain recency of dates based on NBA Schedule
        #Since the season starts in October, 10 is the earliest number.
        #Each "single-digit" month is then compounded with a multiplier to be the most recent 
        #when sorted in descending order
        if int(col.split('-')[0]) < 10:
            multipler = 1
            if int(col.split('-')[0]) == 2:
                multipler = 10
            if int(col.split('-')[0]) == 3:
                multipler = 100
            if int(col.split('-')[0]) == 4:
                multipler = 1000
            if int(col.split('-')[0]) == 5:
                multipler = 10000
            col = str(int(col.split('-')[0]) * 10000 * multipler)+ col.split('-')[1]
        else:
            if int(col.split('-')[1]) < 10: 
                col = col.split('-')[0] + '0' + col.split('-')[1]
            else:
                col = col.split('-')[0] + col.split('-')[1]
        return int(col)