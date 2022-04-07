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
        points = np.vstack(
            (gdata['recall'], gdata['time_total_s'])
        ).transpose()
        params = np.array(gdata['params'])
        if points.shape[0] >= 2:
            print(params)
            points = np.concatenate((np.array([[1, 0]]), points), axis=0)
            # with qhull_options='QG0' we instruct the function to
            # compute the convex hull ignoring the first point, which
            # is somehow the 'observation point'. The field `chull.good`
            # will contain the points of the convex hull which are
            # "visible" from said observation point
            chull = ConvexHull(points, qhull_options='QG0')
            pareto_front = points[chull.vertices[chull.good]]
            pareto_recalls = pareto_front[:,0]
            pareto_times = pareto_front[:,1]
            pareto_params = params[chull.vertices[chull.good] - 1] # offset by 1 because we don't append a dummy point in front of the `params` array
        else:
            pareto_recalls = points[:,0]
            pareto_times = points[:,1]
            pareto_params = params
        return pd.DataFrame({
            'recall': pareto_recalls,
            'time_total_s': pareto_times,
            'params': pareto_params
        })

    data = pd.read_sql("select dataset, workload, k, algorithm, params, threads, recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s from main;", get_db())
    pareto = data.groupby(['dataset', 'workload', 'k', 'algorithm', 'threads']).apply(compute_pareto)
    return pareto.reset_index()


def plot_local_topk():
    data = pd.read_sql("select dataset, workload, k, algorithm, params, threads, recall, time_index_s, time_join_s, time_index_s + time_join_s as time_total_s from main;", get_db())
    # data = get_pareto()
    chart = alt.Chart(data).mark_point(filled=True).encode(
        x=alt.X('recall', type='quantitative', scale=alt.Scale(domain=(0, 1))),
        y=alt.Y('time_total_s', type='quantitative', scale=alt.Scale(type='log')),
        color='algorithm:N'
    ).properties(
        width=1000,
        height=600
    )
    chart.save(os.path.join(BASE_DIR, "plot.html"))


if __name__ == "__main__":
    plot_local_topk()

