from datetime import datetime

OPTI_START_DATE = "2007.01.01" #YYYY.MM.DD
OPTI_END_DATE = "2020.01.01" #YYYY.MM.DD
brick = 12
brick_IS = brick * 4
brick_OOS = brick
brick_REAL = brick
init_dates_list=[]

def split_date(date):
    date_to_split = date.split('.')
    date_year = int(date_to_split[0])
    date_month = int(date_to_split[1])
    date_day = int(date_to_split[2])
    date_list=[date_year,date_month,date_day]
    return(date_list)

def count_months(startdate=OPTI_START_DATE,enddate=OPTI_END_DATE):
    startdate=split_date(OPTI_START_DATE)
    enddate=split_date(OPTI_END_DATE)

    count_months_year = (enddate[0]-startdate[0])*12
    count_months_months = (enddate[1]-startdate[1])
    total_months = count_months_year + count_months_months
    print('years in months',count_months_year)
    print('months diff between dates',count_months_months)
    print('Total months difference:',total_months)
    return total_months

def cast_list_to_string_date(date):
    return str(date[0]) + '.' + str(date[1]) + '.' + str(date[2])

def add_init_cuts(start=OPTI_START_DATE,brick=brick,brick_IS=brick_IS,brick_OOS=brick_OOS,brick_REAL=brick_REAL,end=OPTI_END_DATE):
    start = split_date(OPTI_START_DATE)
    def add_months_Start(brickindex=0,brickvalue=brick,date=start):
        bricks=brickvalue*brickindex
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        date = [year_added,month_added,1]
        return date

    def add_months_IS(brickvalue=brick_IS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_OOS(brickvalue=brick_OOS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    def add_months_Real(brickvalue=brick_OOS, date=start):
        bricks = brickvalue
        brick_year = bricks // 12
        year_added = date[0] + brick_year
        brick_month = bricks % 12
        month_added = date[1] + brick_month
        if month_added >= 13:
            month_added = month_added - 12
            year_added = year_added + 1
        date = [year_added, month_added, 1]
        return date

    iterator = 0
    max_iterations = (count_months()-brick_IS + brick)/brick
    print('This is the number of steps trought the WF: ',round(max_iterations,ndigits=0)-1)
    print('This is what the iteration list would look like:')

    lists = []
    for i in range(int(max_iterations-1)):
        iterator += 1
        the_start_date = add_months_Start(i)
        the_IS_date = add_months_IS(date=the_start_date)
        the_OOS_date = add_months_OOS(date=the_IS_date)
        lists.append([
            cast_list_to_string_date(the_start_date), 
            cast_list_to_string_date(the_IS_date), 
            cast_list_to_string_date(the_OOS_date),
        ])

        print('start date',the_start_date,'forward date ',the_IS_date,'end date ',the_OOS_date)
    print('Max iterations are', iterator)

    return lists