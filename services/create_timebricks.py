from datetime import datetime

class CreateTimebricks:
    """Creates time steps for iterations on wf or data analysis"""
    def __init__(self,start_date, time_step, IS_steps, OOS_steps, REAL_steps, end_date):
        self.start_date = start_date
        self.time_step = time_step
        self.end_date = end_date
        self.IS_steps = IS_steps
        self.OOS_steps = OOS_steps
        self.REAL_steps = REAL_steps

    def split_date(self):
        """Splits string date to integers and returns a tuple"""
        date_to_split = self.split('.')
        date_year = int(date_to_split[0])
        date_month = int(date_to_split[1])
        date_day = int(date_to_split[2])
        date_list = [date_year, date_month, date_day]
        return (date_list)

    def count_months(self):
        """Counts months between the start and end date"""
        startdate = CreateTimebricks.split_date(self.start_date)
        enddate = CreateTimebricks.split_date(self.end_date)
        count_months_year = (enddate[0] - startdate[0]) * 12
        count_months_months = (enddate[1] - startdate[1])
        total_months = count_months_year + count_months_months
        return total_months

    def cast_list_to_string_date(date):
        """Returns a given tuple date to a string"""
        return str(date[0]) + '.' + str(date[1]) + '.' + str(date[2])

    def add_months_start(self, brickindex, date):
        """Adds a step brick value to the start date list"""
        bricks = self.time_step * brickindex
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        date = [year_added, month_added, 1]
        return date

    def add_months_IS(self, date):
        """Adds a step brick value to the IS forward date list"""
        bricks = self.IS_steps
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_OOS(self, date):
        """Adds a step brick value to the OOS forward date list"""
        bricks = self.OOS_steps
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_REAL(self, date):
        """Adds a step brick value to the OOS forward end date list as if
         the execution was made in real account, to prevent forward look bias"""
        bricks = self.REAL_steps
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
        start_on_date = CreateTimebricks.split_date(self.start_date)
        iterator = 0
        max_iterations = (CreateTimebricks.count_months(self) - self.IS_steps + self.time_step) / self.time_step
        print('This is the iteration list:')
        lists = []
        for i in range(int(max_iterations - 1)):
            the_start_date = CreateTimebricks.add_months_start(self,iterator,start_on_date)
            the_IS_date = CreateTimebricks.add_months_IS(self,the_start_date)
            the_OOS_date = CreateTimebricks.add_months_OOS(self,the_IS_date)
            lists.append([
                CreateTimebricks.cast_list_to_string_date(the_start_date),
                CreateTimebricks.cast_list_to_string_date(the_IS_date),
                CreateTimebricks.cast_list_to_string_date(the_OOS_date),
            ])
            iterator += 1
            print('start date', the_start_date, 'forward date ', the_IS_date, 'end date ', the_OOS_date)
        print('Max iterations is', iterator)
        return lists
