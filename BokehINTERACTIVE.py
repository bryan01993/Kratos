from bokeh.layouts import  widgetbox, row , column
from bokeh.models import Slider, ColumnDataSource, Select, CustomJS, ColorBar,LogColorMapper, FixedTicker,LogTicker, ContinuousTicker, BasicTicker ,AdaptiveTicker, ContinuousColorMapper, ColorMapper , LinearColorMapper
from bokeh.plotting import figure, show
from bokeh.io import curdoc, output_notebook
from bokeh.palettes import Spectral5, Inferno256
import pandas as pd
import subprocess

# Slide Lists
BotName = 'EA-B1v2'
i = 'GBPJPY'
j = 'M15'
SIZES = list(range(6, 28, 3))
COLORS = Inferno256
N_SIZES = len(SIZES)
N_COLORS = len(COLORS)
base_dir = 'C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/Optimizaciones Listas/Data encadenada'
first_sequester = base_dir + '/' + 'EA-B1v2 on USDJPY on M15.csv'
df = pd.read_csv(
    first_sequester)

df = df.drop(df[df['Result'] <= 0].index,inplace=True)  # drops custom values below 0

# Execution
def Interactive_Graph(BotName,df,i,j):
    print('Interactive Graph for Phase 1 for {} on {} at {}'.format(BotName,i,j))
    df = pd.read_csv(
        first_sequester)
    df.drop(df[df['Result'] <= 0].index, inplace=True)  # drops custom values below 0
    df.drop(df[df['CustomForward'] <= 0].index, inplace=True)
    df.drop(df[df['Trades'] >=600].index, inplace=True)
    df.to_csv('C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028/reports/EA-B1v2/GBPJPY/M15/Bokehpassed.csv')
    columns = sorted(df.columns)
    discrete = [x for x in columns if df[x].dtype == object]
    continuous = [x for x in columns if x not in discrete]


    def create_figure():
        print('this is Create Figure on Bokeh Interactive for a single chart on unfiltered Phase 1')
        xs = df[x.value].values
        ys = df[y.value].values
        x_title = x.value.title()
        y_title = y.value.title()

        kw = dict()
        if x.value in discrete:
            kw['x_range'] = sorted(set(xs))
        if y.value in discrete:
            kw['y_range'] = sorted(set(ys))
        kw['title'] = "%s vs %s" % (x_title, y_title) + " for {} on {} and {}".format(BotName, i, j)

        p = figure(plot_height=900, plot_width=1700, tools='pan,box_zoom,hover,reset,lasso_select', **kw)
        p.xaxis.axis_label = x_title
        p.yaxis.axis_label = y_title

        if x.value in discrete:
            p.xaxis.major_label_orientation = pd.np.pi / 4

        sz = 9
        if size.value != 'None':
            if len(set(df[size.value])) > N_SIZES:
                groups = pd.qcut(df[size.value].values, N_SIZES, duplicates='drop')
            else:
                groups = pd.Categorical(df[size.value])
            sz = [SIZES[xx] for xx in groups.codes]

        c = "#31AADE"
        if color.value != 'None':
            if len(set(df[color.value])) > N_COLORS:
                groups = pd.qcut(df[color.value].values, N_COLORS, duplicates='drop')
            else:
                groups = pd.Categorical(df[color.value])
            c = [COLORS[xx] for xx in groups.codes]

        Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(df['Profit']),high=max(df['Profit']))  # arreglar Maximo y minimo para que agarren el valor
        #Var_color_mapper = LinearColorMapper(palette="Inferno256",low=min(df[color.value]),high=max(df[color.value]))  # arreglar Maximo y minimo para que agarren el valor
        GraphTicker = AdaptiveTicker(base=50,desired_num_ticks=10,num_minor_ticks=20,max_interval=1000)
        Color_legend = ColorBar(color_mapper=Var_color_mapper,ticker =GraphTicker,label_standoff=12, border_line_color=None,location=(0, 0)) #arreglar LogTicker para que muestre por al escala del color
        p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.1, hover_color='white', hover_alpha=0.1)
        p.add_layout(Color_legend,'right')
        p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.1, hover_color='white', hover_alpha=0.1)
        return p


    def callback(attr, old, new):
        print('entrando a callback')
        layout.children[1] = create_figure()
        callback = CustomJS(code="console.log('tap event occurred')")
        print('actualizado y saliendo de callback')

    source = ColumnDataSource(data=dict(x=df['Pass'], y=df['Profit']))
    x = Select(title='X-Axis', value='Pass', options=columns)
    x.on_change('value', callback)
    y = Select(title='Y-Axis', value='Profit', options=columns)
    y.on_change('value', callback)
    size = Select(title='Size', value='None', options=['None'] + continuous)
    size.on_change('value', callback)
    color = Select(title='Color', value='None', options=['None'] + continuous)
    color.on_change('value', callback)

    controls = column(y, x, color, size, width=200)
    layout = row(controls, create_figure())

    curdoc().add_root(layout)
    #show(layout)
    output_notebook()
    curdoc().title = "Phase 1 for {} on {} at {}".format(BotName,i,j),

    process = subprocess.call('bokeh serve --show BokehINTERACTIVE.py')
Interactive_Graph(BotName,df,i,j)
    # bokeh serve - -show bokehINTERACTIVE.py
