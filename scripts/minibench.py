# Runs a simple indexing and query, recording the execution time in
# a simple database. Useful to detect performance regressions.
#
import puffinn
import time
import h5py
import sqlite3
import subprocess as sp
from datetime import datetime

MB = 1024 * 1024


def run(data_path):
    print("Running experiment")
    k = 10
    with h5py.File(data_path, "r") as hfp:
        train = hfp["train"][:]
        test = hfp["test"][:]

    n, dimensions = train.shape
    n_queries = test.shape[0]

    print("  building index")
    t_start = time.time()
    index = puffinn.Index("angular", dimensions, 4 * 1024 * MB)
    for v in train:
        v = list(v)
        index.insert(v)
    index.rebuild()
    t_index_secs = time.time() - t_start

    print("  running queries")
    t_start = time.time()
    for q in test:
        q = list(q)
        index.search(q, k, 0.9)
    t_query_secs = time.time() - t_start
    return {
        "time_index_s": t_index_secs,
        "time_query_s": t_query_secs,
        "index_pps": n / t_index_secs,
        "qps": n_queries / t_query_secs,
    }


def get_git_info():
    p = sp.run(["git", "rev-parse", "HEAD"], capture_output=True)
    sha = p.stdout.strip().decode("utf-8")
    p = sp.run(["git", "show", "-s", "--format=%ci", "HEAD"], capture_output=True)
    date = p.stdout.strip().decode("utf-8")
    p = sp.run(["git", "diff"], capture_output=True)
    diff = p.stdout.decode("utf-8")
    return {
        "git_commit": sha,
        "git_diff": diff,
        "git_date": date,
    }


def get_db():
    conn = sqlite3.connect(".minibench.db")

    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS knn_queries (
            data_path     TEXT,
            git_commit    TEXT,
            git_diff      TEXT,
            git_date      DATETIME,
            run_date      DATETIME,
            time_index_s   REAL,
            time_query_s   REAL,
            index_pps      REAL,
            qps            REAL,
            PRIMARY KEY(git_commit, git_diff, data_path)
        );
        """
    )

    return conn


def already_run(db, git_info, data_path):
    params = dict(git_info)
    params["data_path"] = data_path
    (res,) = db.execute(
        """SELECT COUNT(*) FROM knn_queries 
           WHERE git_commit=:git_commit 
           AND git_diff=:git_diff 
           AND data_path=:data_path""",
        params,
    ).fetchone()
    return res > 0


if __name__ == "__main__":
    path = "glove-100-angular.hdf5"
    git_info = get_git_info()
    with get_db() as db:
        if already_run(db, git_info, path):
            yn = ""
            while yn.lower() not in ["y", "n"]:
                yn = input(
                    "Benchmark already run in this configuration. Overwrite? [y/n] "
                )
                if yn == "n":
                    exit("Doing nothing")
        exec_info = run(path)
        exec_info["data_path"] = path
        exec_info["run_date"] = datetime.now().isoformat()
        exec_info.update(git_info)
        db.execute(
            """
            INSERT OR REPLACE INTO knn_queries (
                data_path     ,
                git_commit    ,
                git_diff      ,
                git_date      ,
                run_date      ,
                time_index_s  ,
                time_query_s  ,
                index_pps     ,
                qps
            ) VALUES (
                :data_path     ,
                :git_commit    ,
                :git_diff      ,
                :git_date      ,
                :run_date      ,
                :time_index_s  ,
                :time_query_s  ,
                :index_pps     ,
                :qps
            )
        """,
            exec_info,
        )
