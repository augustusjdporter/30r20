from ci import predband
import matplotlib.pyplot as plt
import datetime
import numpy as np
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.api as sm
import uncertainties.unumpy as unp
import uncertainties as unc
from strava_api import get_strava_data
from logbook_api import get_logbook_data

def convert_numeric_date_to_date(numeric_date):
    return numeric_date * pd.Timedelta("1d") + pd.Timestamp("1970-01-01")

def convert_date_to_numeric_date(date):
    return (date - pd.Timestamp("1970-01-01")) / pd.Timedelta("1d")

def get_distance(split_string):
    [mins, secs] = split_string.split(":")
    mins = float(mins)
    secs = float(secs)
    secs += (mins * 60.)
    pace = secs / 500.
    speed = 1/pace
    time = 30 * 60
    distance = speed * time
    print(split_string, distance)
    return distance


def get_data(data_source):
    if data_source == "logbook":
        data = get_logbook_data()
    elif data_source == "strava":
        data = get_strava_data()
    else:
        raise Exception(f"unsupported data source: {data_source}")
    dates = pd.to_datetime(data["date"]).dt.tz_localize(None)
    numeric_dates = convert_date_to_numeric_date(dates)
    distances = data["distance"]

    return np.array(dates), np.array(numeric_dates), np.array(distances)

def run(goal_split = "1:48.8", y_min=7900, y_max=8300, data_source="logbook"):
    dates, numeric_dates, distances = get_data(data_source=data_source)
    if len(distances) < 3:
        raise Exception("Need at least 3 data points")

    indices = np.argsort(numeric_dates)
    numeric_dates = numeric_dates[indices]
    distances = distances[indices]
    dates = dates[indices]

    if len(numeric_dates) < 3:
        return

    plt.figure(figsize=(6.5, 5))

    min_num_date = np.min(numeric_dates)
    numeric_dates -= min_num_date

    X = numeric_dates.reshape(-1, 1)

    X = sm.add_constant(X)

    model = sm.OLS(distances, X)
    results = model.fit()
    predict = results.predict(X)
    print(results.conf_int(0.01))
    print(results.summary())
    print("pp vals", results.pvalues)
    fit_vals = results.params
    r_sq = results.rsquared
    error = results.bse

    plt.errorbar(dates, distances, fmt="o", zorder=1000)
    plt.plot(dates, predict, label=r"fit: distance = $({:.2f}\pm{:.2f})\, d + ({:.2f}\pm{:.2f})$"
              .format(fit_vals[1], error[1], fit_vals[0], error[0]))
    plt.scatter(dates[0], np.median(distances), c="white", zorder=-999, label=r"$R^2$: {:.3f}, $p$: {:.4f}".format(r_sq,
                                                                                                               results.pvalues[1]))
    plt.xlabel("Date")
    plt.ylabel("Distance")

    numeric_date_grid = np.linspace(0, np.max(numeric_dates)-np.min(numeric_dates)+0*100, 10 ** 3)
    numeric_date_invisible = np.linspace(0, 10**4, 3*10 ** 4)
    new_dates = []
    for numeric_date_item in numeric_date_grid:
        date = convert_numeric_date_to_date(min_num_date + numeric_date_item)
        d = date.to_pydatetime()
        new_dates.append(d)

    X = numeric_date_grid.reshape(-1, 1)
    X = sm.add_constant(X)

    def f(x,b,a):
        return x*a + b
    error_preds = predband(numeric_date_grid, numeric_dates, distances, fit_vals, f)

    pcov = results.cov_params()
    # because b and a are correlated, cannot do a simple adding errors in quadrature
    b, a = unc.correlated_values(fit_vals, pcov)
    predict = a*numeric_date_grid + b
    e_mean = unp.std_devs(predict)
    predict = unp.nominal_values(predict)

    y_desired = get_distance(goal_split)

    predict_invisible = a * numeric_date_invisible + b
    e_inv = unp.std_devs(predict_invisible)
    predict_invisible = unp.nominal_values(predict_invisible)
    factor = 1.96
    above = predict_invisible + factor*e_inv
    below = predict_invisible - factor*e_inv

    index = np.argmin(np.abs(above - y_desired))
    t_above = numeric_date_invisible[index]

    index = np.argmin(np.abs(below - y_desired))
    t_below = numeric_date_invisible[index]
    below_value = below[index]

    min_date = convert_numeric_date_to_date(min_num_date + t_above)
    max_date = convert_numeric_date_to_date(min_num_date + t_below)
    if below_value < 8175:
        max_date = ["-", "-", "-"]

    plt.axhline(y_desired, c="gold")

    interval = 1.96 * e_mean
    plt.plot(new_dates, predict + interval, c="black", label=r"Goal time: {} - {} (95% CL)".format(min_date, max_date))
    plt.plot(new_dates, predict - interval, c="black")
    plt.plot(new_dates, predict + error_preds, c="red", label="95% PL")
    plt.plot(new_dates, predict - error_preds, c="red")
    y_lims = plt.ylim()
    if y_lims[1] > 210:
        y_lims = (y_lims[0], 210)
    if y_lims[0] < 145:
        y_lims = (145, y_lims[1])

    plt.gca().xaxis.set_major_locator(mticker.MaxNLocator(nbins=6))
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
               ncol=1, mode="expand", borderaxespad=0., title="30r20 distance")

    orig_ax = plt.gca()
    ax = plt.gca().twiny().twinx()
    ax.set_xticks([])
    ax.set_yticks(orig_ax.get_yticks())


    y_tick_labels = ["1:48", "1:49", "1:50", "1:51", "1:52", "1:53"]

    y_ticks = []
    for y_tick_label in y_tick_labels:
        dist_labels = get_distance(y_tick_label)
        y_ticks.append(dist_labels)
        ax.axhline(dist_labels, c="gray", linestyle="--", alpha=0.2)
        orig_ax.axhline(dist_labels, c="gray", linestyle="--", alpha=0.0000001)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_tick_labels)
    ax.set_ylabel("Split$\,$/$\,$500$\,$m")

    orig_ax.set_ylim([y_min, y_max])
    ax.set_ylim([y_min, y_max])

    x_lim_window = 10
    min_date = pd.Timestamp(min(dates))
    max_date = pd.Timestamp(max(dates))
    orig_ax.set_xlim([min_date - datetime.timedelta(days=x_lim_window), max_date + datetime.timedelta(days=x_lim_window)])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run()