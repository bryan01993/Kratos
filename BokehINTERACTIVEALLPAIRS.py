from bokeh.layouts import  widgetbox, row , column, gridplot, grid
from bokeh.models import Slider, ColumnDataSource, Select, CustomJS, ColorBar,LogColorMapper, FixedTicker,LogTicker, ContinuousTicker, BasicTicker ,AdaptiveTicker, ContinuousColorMapper, ColorMapper , LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.io import curdoc, output_notebook
from bokeh.palettes import Spectral5, Inferno256
import pandas as pd
import subprocess
# THIS SHALL BE Graph Phase 2 Button

# Slide Lists
BotName = 'EA-TR1v1'
EURUSD = 'EURUSD'
GBPUSD = 'GBPUSD'
USDCAD = 'USDCAD'
USDCHF = 'USDCHF'
USDJPY = 'USDJPY'
TimeFrame = 'M5'
Phase='1'
SIZES = list(range(6, 28, 3))
COLORS = Inferno256
N_SIZES = len(SIZES)
N_COLORS = len(COLORS)

dfp1 = pd.read_csv(
    'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
        BotName, EURUSD, TimeFrame, BotName, EURUSD, TimeFrame,Phase))
# Execution
def Interactive_Graph():
    print('Interactive Graph for Phase {} for {} on All Pairs at {}'.format(Phase,BotName,TimeFrame))
    dfp1 = pd.read_csv(
        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
            BotName, EURUSD, TimeFrame, BotName, EURUSD, TimeFrame,Phase))
    dfp2 = pd.read_csv(
        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
            BotName, GBPUSD, TimeFrame, BotName, GBPUSD, TimeFrame,Phase))

    dfp3 = pd.read_csv(
        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
            BotName, USDCAD, TimeFrame, BotName, USDCAD, TimeFrame,Phase))

    dfp4 = pd.read_csv(
        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
            BotName, USDCHF, TimeFrame, BotName, USDCHF, TimeFrame,Phase))

    dfp5 = pd.read_csv(
        'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/{}/{}/{}/OptiResults-{}-{}-{}-Phase{}.csv'.format(
            BotName, USDJPY, TimeFrame, BotName, USDJPY, TimeFrame,Phase))

    columns = sorted(dfp1.columns)
    discrete = [x for x in columns if dfp1[x].dtype == object]
    continuous = [x for x in columns if x not in discrete]


    def create_figure1():
        xsp1 = dfp1[x.value].values
        ysp1 = dfp1[y.value].values
        x_titlep1 = x.value.title()
        y_titlep1 = y.value.title()

        kwp1 = dict()
        if x.value in discrete:
            kwp1['x_range'] = sorted(set(xsp1))
        if y.value in discrete:
            kwp1['y_range'] = sorted(set(ysp1))
        kwp1['title'] = "%s vs %s" % (x_titlep1, y_titlep1) + " for {} on {} and {}".format(BotName, EURUSD, TimeFrame)

        pp1 = figure(plot_height=400, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kwp1)
        pp1.xaxis.axis_label = x_titlep1
        pp1.yaxis.axis_label = y_titlep1

        if x.value in discrete:
            pp1.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(dfp2[size.value])) > N_SIZES:
                groups = pd.qcut(dfp2[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(dfp2[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(dfp2[color.value])) > N_COLORS:
                groups = pd.qcut(dfp2[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(dfp2[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        # COLOR BAR NEXT TO GRAPHIC

        #PAIR 1
        try:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1['Profit']),high=max(dfp1['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        except ValueError:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=0,high=1)
            print('This {} did not launch Phase {} on {}'.format(BotName,Phase,TimeFrame))

        #Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1[color.value]),high=max(dfp1[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50,desired_num_ticks=10,num_minor_ticks=20,max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper,ticker =GraphTicker,label_standoff=12, border_line_color=None,location=(0, 0)) #arreglar LogTicker para que muestre por al escala del color
        pp1.circle(x=xsp1, y=ysp1, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white', hover_alpha=0.5)
        pp1.add_layout(Color_legend,'right')

        return pp1

    def create_figure2():
        xsp2 = dfp2[x.value].values
        ysp2 = dfp2[y.value].values
        x_titlep2 = x.value.title()
        y_titlep2 = y.value.title()

        kwp2 = dict()
        if x.value in discrete:
            kwp2['x_range'] = sorted(set(xsp2))
        if y.value in discrete:
            kwp2['y_range'] = sorted(set(ysp2))
        kwp2['title'] = "%s vs %s" % (x_titlep2, y_titlep2) + " for {} on {} and {}".format(BotName, GBPUSD, TimeFrame)

        pp2 = figure(plot_height=400, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kwp2)
        pp2.xaxis.axis_label = x_titlep2
        pp2.yaxis.axis_label = y_titlep2

        if x.value in discrete:
            pp2.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(dfp2[size.value])) > N_SIZES:
                groups = pd.qcut(dfp2[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(dfp2[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(dfp2[color.value])) > N_COLORS:
                groups = pd.qcut(dfp2[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(dfp2[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        # COLOR BAR NEXT TO GRAPHIC
        #PAIR 2
        try:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1['Profit']),high=max(dfp1['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        except ValueError:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=0,high=1)
            print('This {} did not launch Phase {} on {}'.format(BotName,Phase,TimeFrame))
        # Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1[color.value]),high=max(dfp1[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50, desired_num_ticks=10, num_minor_ticks=20, max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper, ticker=GraphTicker, label_standoff=12,border_line_color=None,location=(0, 0))  # arreglar LogTicker para que muestre por al escala del color
        pp2.circle(x=xsp2, y=ysp2, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',hover_alpha=0.5)
        pp2.add_layout(Color_legend, 'right')
        return pp2

    def create_figure3():
        xsp3 = dfp3[x.value].values
        ysp3 = dfp3[y.value].values
        x_titlep3 = x.value.title()
        y_titlep3 = y.value.title()

        kwp3 = dict()
        if x.value in discrete:
            kwp3['x_range'] = sorted(set(xsp3))
        if y.value in discrete:
            kwp3['y_range'] = sorted(set(ysp3))
        kwp3['title'] = "%s vs %s" % (x_titlep3, y_titlep3) + " for {} on {} and {}".format(BotName, USDCAD, TimeFrame)

        pp3 = figure(plot_height=400, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kwp3)
        pp3.xaxis.axis_label = x_titlep3
        pp3.yaxis.axis_label = y_titlep3

        if x.value in discrete:
            pp3.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(dfp3[size.value])) > N_SIZES:
                groups = pd.qcut(dfp3[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(dfp3[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(dfp3[color.value])) > N_COLORS:
                groups = pd.qcut(dfp3[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(dfp3[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        # COLOR BAR NEXT TO GRAPHIC
        #PAIR 3
        try:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1['Profit']),high=max(dfp1['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        except ValueError:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=0,high=1)
            print('This {} did not launch Phase {} on {}'.format(BotName,Phase,TimeFrame))
        # Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1[color.value]),high=max(dfp1[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50, desired_num_ticks=10, num_minor_ticks=20, max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper, ticker=GraphTicker, label_standoff=12,border_line_color=None,location=(0, 0))
        pp3.circle(x=xsp3, y=ysp3, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',hover_alpha=0.5)
        pp3.add_layout(Color_legend, 'right')
        return pp3

    def create_figure4():
        xsp4 = dfp4[x.value].values
        ysp4 = dfp4[y.value].values
        x_titlep4 = x.value.title()
        y_titlep4 = y.value.title()

        kwp4 = dict()
        if x.value in discrete:
            kwp4['x_range'] = sorted(set(xsp4))
        if y.value in discrete:
            kwp4['y_range'] = sorted(set(ysp4))
        kwp4['title'] = "%s vs %s" % (x_titlep4, y_titlep4) + " for {} on {} and {}".format(BotName, USDCHF, TimeFrame)

        pp4 = figure(plot_height=400, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kwp4)
        pp4.xaxis.axis_label = x_titlep4
        pp4.yaxis.axis_label = y_titlep4

        if x.value in discrete:
            pp4.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(dfp4[size.value])) > N_SIZES:
                groups = pd.qcut(dfp4[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(dfp4[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(dfp4[color.value])) > N_COLORS:
                groups = pd.qcut(dfp4[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(dfp4[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        # COLOR BAR NEXT TO GRAPHIC
        #PAIR 4
        try:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1['Profit']),high=max(dfp1['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        except ValueError:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=0,high=1)
            print('This {} did not launch Phase {} on {}'.format(BotName,Phase,TimeFrame))
        # Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1[color.value]),high=max(dfp1[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50, desired_num_ticks=10, num_minor_ticks=20, max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper, ticker=GraphTicker, label_standoff=12,border_line_color=None,location=(0, 0))
        pp4.circle(x=xsp4, y=ysp4, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',hover_alpha=0.5)
        pp4.add_layout(Color_legend, 'right')
        return pp4

    def create_figure5():
        xsp5 = dfp5[x.value].values
        ysp5 = dfp5[y.value].values
        x_titlep5 = x.value.title()
        y_titlep5 = y.value.title()

        kwp5 = dict()
        if x.value in discrete:
            kwp5['x_range'] = sorted(set(xsp5))
        if y.value in discrete:
            kwp5['y_range'] = sorted(set(ysp5))
        kwp5['title'] = "%s vs %s" % (x_titlep5, y_titlep5) + " for {} on {} and {}".format(BotName, USDJPY, TimeFrame)

        pp5 = figure(plot_height=500, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kwp5)
        pp5.xaxis.axis_label = x_titlep5
        pp5.yaxis.axis_label = y_titlep5

        if x.value in discrete:
            pp5.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(dfp5[size.value])) > N_SIZES:
                groups = pd.qcut(dfp5[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(dfp5[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(dfp5[color.value])) > N_COLORS:
                groups = pd.qcut(dfp5[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(dfp5[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        # COLOR BAR NEXT TO GRAPHIC
        #PAIR 5
        try:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1['Profit']),high=max(dfp1['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        except ValueError:
            Var_color_mapper = LinearColorMapper(palette="Inferno256",low=0,high=1)
            print('This {} did not launch Phase {} on {}'.format(BotName,Phase,TimeFrame))
        # Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(dfp1[color.value]),high=max(dfp1[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50, desired_num_ticks=10, num_minor_ticks=20, max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper, ticker=GraphTicker, label_standoff=12,border_line_color=None,location=(0, 0))
        pp5.circle(x=xsp5, y=ysp5, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',hover_alpha=0.5)
        pp5.add_layout(Color_legend, 'right')
        return pp5

    def callback(attr, old, new):

        layout.children[1] = grid([create_figure1(), create_figure2(), create_figure3(),create_figure4(),create_figure5()], ncols=2)

        callback = CustomJS(code="console.log('tap event occurred')")

    #source = ColumnDataSource(data=dict(x=dfp1['Pass'], y=dfp1['Profit']))
    x = Select(title='X-Axis', value='Pass', options=columns)
    x.on_change('value', callback)

    y = Select(title='Y-Axis', value='Profit', options=columns)
    y.on_change('value', callback)

    size = Select(title='Size', value='None', options=['None'] + continuous)
    size.on_change('value', callback)

    color = Select(title='Color', value='None', options=['None'] + continuous)
    color.on_change('value', callback)

    controls = column(y, x, color, size, width=200)
    #layout = row(controls, create_figure1(),create_figure2(),create_figure3()) # ESTE FUNCIONA GUARDAR POR AHORA
    #layoutgrid = gridplot([controls,create_figure1(),create_figure2(),create_figure3()],toolbar_location='left',sizing_mode='stretch_both',ncols=2) # ESTO NO ACTUALIZA RESULTADOS A PESAR DE QUE SI LOS UBICA CORRECTAMENTE
    #layout = row(controls, layoutgrid)
    layoutgrid = grid([create_figure1(), create_figure2(), create_figure3(),create_figure4(),create_figure5()], ncols=2) #ESTE SI FUNCIONA PERO AL ACTUALIZAR RESULTADOS DEJA SOLO FIGURA 1
    layout = row(controls,layoutgrid)
    curdoc().add_root(layout)
    output_notebook()
    curdoc().title = "Phase {} All Pairs on {}".format(Phase, TimeFrame),

    process = subprocess.call('bokeh serve --show BokehINTERACTIVEALLPAIRS.py')
Interactive_Graph()



"""slider_widget1 = Slider(start= 0, end = 100, step = 10, title = 'Slider 1')

slider_widget2 = Slider(start= 0, end = 20, step = 1, title = 'Slider 2')

slider_widget3 = Slider(start= 0, end = 200, step = 5, title = 'Slider 3')

slider_layout = widgetbox(slider_widget1,slider_widget2,slider_widget3)

curdoc().add_root(slider_layout)"""

#Exercise 2

"""initial_points = 500
data_points = ColumnDataSource(data = {'x':random(initial_points),'y':random(initial_points)})

plot = figure(title = 'Random Scatter Plot Generator')

plot.diamond(x = 'x', y = 'y', source = data_points, color = 'red')

select_widget = Select(options = ['Uniform Distribution','Normal Distribution'], value = 'Uniform Distribution', title = 'Select Distribution')

slider_widget = Slider(start= 0, end = 10000, step = 10, value = initial_points, title = 'Slide right to increase n of points')

def callback(attr,old,new):

    if select_widget.value == 'Uniform Distribution':
        function = random
    else:
        function = normal

    points = slider_widget.value
    data_points.data = {'x':function(random(points)),'y':function(random(points))}

slider_widget.on_change('value',callback)

select_widget.on_change('value',callback)

layout = row(slider_widget,select_widget, plot)

curdoc().add_root(layout)"""