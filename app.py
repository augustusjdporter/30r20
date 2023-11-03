from ci import predband
import matplotlib.pyplot as plt
import datetime
import numpy as np
import jdcal
import matplotlib.ticker as mticker
import pandas as pd
import statsmodels.api as sm
import uncertainties.unumpy as unp
import uncertainties as unc

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


def get_data(file):
    data = pd.read_csv(file)
    dates = []
    jds = []
    distances = []
    # for distance, day, month, year in data:
    for index, row in data.iterrows():
        distance, day, month, year = row
        distances.append(distance)
        jd = sum(jdcal.gcal2jd(year, month, day))
        date = datetime.date(year, month, day)
        jds.append(jd)
        dates.append(date)


    return np.array(dates), np.array(jds), np.array(distances)
def run(file, goal_split = "1:48.8", y_min=7900, y_max=8300):
    dates, jds, distances = get_data(file=file)
    if len(distances) < 3:
        raise Exception("Need at least 3 data points")

    indices = np.argsort(jds)
    jds = jds[indices]
    distances = distances[indices]
    dates = dates[indices]

    if len(jds) < 3:
        return

    plt.figure(figsize=(6.5, 5))

    min_jds = np.min(jds)
    jds -= min_jds

    X = jds.reshape(-1, 1)

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

    new_jds = np.linspace(0, np.max(jds)-np.min(jds)+0*100, 10 ** 3)
    new_jds_invisible = np.linspace(0, 10**4, 3*10 ** 4)
    new_dates = []
    for njd in new_jds:
        YYYY, MM, DD, HH = jdcal.jd2gcal(min_jds, njd)
        HH *= 24
        mm = (HH % 1) * 60
        ss = (mm % 1) * 60
        HH = int(HH)
        mm = int(mm)
        ss = int(ss)
        d = datetime.datetime(YYYY, MM, DD, HH, mm, ss)
        new_dates.append(d)

    X = new_jds.reshape(-1, 1)
    X = sm.add_constant(X)

    def f(x,b,a):
        return x*a + b
    error_preds = predband(new_jds, jds, distances, fit_vals, f)

    pcov = results.cov_params()
    # because b and a are correlated, cannot do a simple adding errors in quadrature
    b, a = unc.correlated_values(fit_vals, pcov)
    predict = a*new_jds + b
    e_mean = unp.std_devs(predict)
    predict = unp.nominal_values(predict)

    y_desired = get_distance(goal_split)

    predict_invisible = a * new_jds_invisible + b
    e_inv = unp.std_devs(predict_invisible)
    predict_invisible = unp.nominal_values(predict_invisible)
    factor = 1.96
    above = predict_invisible + factor*e_inv
    below = predict_invisible - factor*e_inv

    index = np.argmin(np.abs(above - y_desired))
    t_above = new_jds_invisible[index]

    index = np.argmin(np.abs(below - y_desired))
    t_below = new_jds_invisible[index]
    below_value = below[index]

    min_date = jdcal.jd2gcal(min_jds, t_above)
    max_date = jdcal.jd2gcal(min_jds, t_below)
    if below_value < 8175:
        max_date = ["-", "-", "-"]

    plt.axhline(y_desired, c="gold")

    interval = 1.96 * e_mean
    plt.plot(new_dates, predict + interval, c="black", label=r"Goal time: {}-{}-{} - {}-{}-{} (95% CL)".format(min_date[0], min_date[1], min_date[2], max_date[0], max_date[1], max_date[2]))
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
    orig_ax.set_xlim([min(dates) - datetime.timedelta(days=x_lim_window), max(dates) + datetime.timedelta(days=x_lim_window)])
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run(file="/Users/augustusporter/projects/test_project/30r20/30r20.csv")