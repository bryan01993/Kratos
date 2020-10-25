import pandas as pd
import os.path
from bokeh.plotting import curdoc, figure
from bokeh.models import Select, CustomJS
from bokeh.layouts import row, column
from bokeh.io import output_notebook


FOLDER_PATH = "C:/Users/bryan/AppData/Roaming/MetaQuotes/Terminal/6C3C6A11D1C3791DD4DBF45421BF8028"
REPORT_PATH = os.path.join(FOLDER_PATH, 'reports')

class GraphBacktestPhase2:
    def __init__(self, dto):
        self.bot = dto.bot
        self.pairs = dto.pairs
        self.time_frames = dto.time_frames

    def graph_backtest_phase2(self):
        for pair in self.pairs:
            for time_frame in self.time_frames:
                if self.pairs[pair] == 1 and self.time_frames[time_frame] == 1:
                    self.graph(pair, time_frame)


    def graph(self, pair, time_frame):
        path = os.path.join(REPORT_PATH, self.bot, pair, 'OptiResults-{}-{}-{}-Phase2.csv'.format(self.bot, pair, time_frame))
        df = pd.read_csv(path)

        columns = sorted(df.columns)
        discrete = [x for x in columns if df[x].dtype == object]
        continuous = [x for x in columns if x not in discrete]

        def create_figure():
            xs = df[x.value].values
            ys = df[y.value].values
            x_title = x.value.title()
            y_title = y.value.title()

            kw = dict()
            if x.value in discrete:
                kw['x_range'] = sorted(set(xs))
            if y.value in discrete:
                kw['y_range'] = sorted(set(ys))
            kw['title'] = "%s vs %s" % (x_title, y_title) + " for {} on {} and {}".format(BotName, pair, time_frame)

            p = figure(plot_height=600, plot_width=800, tools='pan,box_zoom,hover,reset,lasso_select', **kw)
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

            p.circle(x=xs, y=ys, color=c, size=sz, line_color="white", alpha=0.6, hover_color='white',
                        hover_alpha=0.5)
            return p

        def callback(attr, old, new):
            layout.children[1] = create_figure()
            callback = CustomJS(code="console.log('tap event occurred')")

        # source = ColumnDataSource(data=dict(x=df['Pass'], y=df['Profit']))
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
        # show(layout)
        output_notebook()
        curdoc().title = "Phase 1 Graph unfiltered for {} on {} at {}".format(self.bot, pair, time_frame)

        process = subprocess.call('bokeh serve --show ProductionOpti.py')
        # Posteriormente agregar lineas del CSV en vivo con sliders tal como en https://demo.bokeh.org/export_csv
        # bokeh serve - -show ProductionOpti.py
