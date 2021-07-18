from datetime import datetime
from .helpers import split_date, cast_list_to_string_date

class CreateTimebricks:
    """Creates time steps for iterations on wf or data analysis"""
    def __init__(self,start_date, time_step, is_steps, oos_steps, real_steps, end_date):
        self.start_date = start_date   # YYYY.MM.DD
        self.time_step = time_step   # in months
        self.end_date = end_date   # YYYY.MM.DD
        self.is_steps = is_steps  # in months
        self.oos_steps = oos_steps  # in months
        self.real_steps = real_steps  # in months

    def count_months(self):
        """Counts months between the start and end date"""
        startdate = split_date(self.start_date)
        enddate = split_date(self.end_date)
        count_months_year = (enddate[0] - startdate[0]) * 12
        count_months_months = (enddate[1] - startdate[1])
        total_months = count_months_year + count_months_months
        return total_months

    def add_months_start(self, brickindex, date):
        """Adds a step brick value to the start date list"""
        bricks = self.time_step * brickindex
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        date = [year_added, month_added, 1]
        return date

    def add_months_is(self, date):
        """Adds a step brick value to the is forward date list"""
        bricks = self.is_steps
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_oos(self, date):
        """Adds a step brick value to the oos forward date list"""
        bricks = self.oos_steps
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_real(self, date):
        """Adds a step brick value to the oos forward end date list as if
         the execution was made in real account, to prevent forward look bias"""
        bricks = self.real_steps
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date


    def run(self):
        """Standard Process of the class"""
        start_on_date = split_date(self.start_date)
        iterator = 0
        max_iterations = (CreateTimebricks.count_months(self) - self.is_steps + self.time_step) / self.time_step
        #print('This is the iteration list:')
        lists = []
        for i in range(int(max_iterations - 1)):
            the_start_date = CreateTimebricks.add_months_start(self,iterator,start_on_date)
            the_is_date = CreateTimebricks.add_months_is(self,the_start_date)
            the_oos_date = CreateTimebricks.add_months_oos(self,the_is_date)
            lists.append([
                cast_list_to_string_date(the_start_date),
                cast_list_to_string_date(the_is_date),
                cast_list_to_string_date(the_oos_date),
            ])
            iterator += 1
            #print('start date', the_start_date, 'forward date ', the_is_date, 'end date ', the_oos_date)
        #print('Max iterations is', iterator)
        return lists

