#!/usr/bin/env python3

import sqlite3
import numpy as np
import altair as alt
import sys
from scipy.spatial import ConvexHull
import os
import pandas as pd

DIR_ENVVAR = 'TOPK_DIR'
try:
    BASE_DIR = os.environ[DIR_ENVVAR]
except:
    print("You should set the {} environment variable to a directory".format(DIR_ENVVAR))
    sys.exit(1)

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
RESULT_FILES_DIR = os.path.join(BASE_DIR, "output")


def get_db():
    db = sqlite3.connect(os.path.join(BASE_DIR, "join-results.db"))
    return db


def get_pareto():
    def compute_pareto(gdata):
        gdata = gdata.sort_values(['time_total_s'], ascending=True)
        points = np.vstack(
            (gdata['recall'], gdata['time_total_s'])
        ).transpose()

        # now we seek the vertices of the pareto 
        # frontier to select from the `gdata` object
        indices = []
        last_r = 0
        for i, (r, t) in enumerate(points):
            if r > last_r:
                last_r = r
                indices.append(i)
        return gdata[['recall', 'time_total_s', 'params']].iloc[indices]

    data = pd.read_sql("select dataset, workload, k, algorithm, params, threads, recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s from main;", get_db())
    pareto = data.groupby(['dataset', 'workload', 'k', 'algorithm', 'threads']).apply(compute_pareto)
    return pareto.reset_index()


def plot_local_topk():
    all = pd.read_sql("select dataset, workload, k, algorithm, params, threads, recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s from main;", get_db())
    data = get_pareto()
    chart_pareto = alt.Chart(data).mark_line(point=True).encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color='algorithm:N',
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q'
        ]
    )
    chart_all = alt.Chart(all).mark_point().encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color='algorithm:N',
        tooltip=[
            'algorithm:N',
            'params:N',
            'recall:Q',
            'time_total_s:Q'
        ]
    )
    
    chart = alt.layer(chart_all, chart_pareto).properties(
        width=1000,
        height=600,
        title="Recall vs. time"
    )
    chart.save(os.path.join(BASE_DIR, "plot.html"))


if __name__ == "__main__":
    plot_local_topk()

